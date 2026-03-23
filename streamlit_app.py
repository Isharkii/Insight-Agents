"""Minimal Streamlit frontend for the modular Insight Agent."""

from __future__ import annotations

import io
import json
import os
import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from pydantic import ValidationError

from llm_synthesis.schema import InsightOutput

st.set_page_config(page_title="InsightAgent", page_icon="IA", layout="wide")
logger = logging.getLogger(__name__)
FinalInsightResponse = InsightOutput
_FINAL_RESPONSE_KEYS = frozenset(FinalInsightResponse.model_fields.keys())

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
_BUSINESS_TYPE_OPTIONS = [
    "auto",
    "saas",
    "ecommerce",
    "agency",
    "general_timeseries",
    "financial_markets",
    "marketing_analytics",
    "operations",
    "retail",
    "healthcare",
    "generic_timeseries",
]
_MULTI_ENTITY_OPTIONS = ["auto", "split", "error"]


@st.cache_data(show_spinner=False)
def _discover_clients() -> list[str]:
    """Discover client IDs from local config files."""
    clients_dir = Path(os.getenv("CLIENT_CONFIG_DIR", "config/clients"))
    clients: list[str] = []
    if clients_dir.exists() and clients_dir.is_dir():
        clients = sorted(p.stem for p in clients_dir.glob("*.json"))
    return ["default", *clients]


@st.cache_data(show_spinner=False)
def _load_local_client_config(client_id: str) -> Optional[dict[str, Any]]:
    """Load local client config JSON if present."""
    if not client_id or client_id == "default":
        return None
    config_path = Path(os.getenv("CLIENT_CONFIG_DIR", "config/clients")) / f"{client_id}.json"
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as fp:
        loaded = json.load(fp)
    return loaded if isinstance(loaded, dict) else None


@st.cache_data(show_spinner=False)
def _load_csv_preview(data: bytes, preview_rows: int) -> tuple[pd.DataFrame, list[str]]:
    """Load bounded CSV preview for memory-friendly display."""
    preview_df = pd.read_csv(io.BytesIO(data), nrows=preview_rows)
    return preview_df, list(preview_df.columns)


def _fetch_remote_client_config_placeholder(client_id: str) -> dict[str, Any]:
    """Placeholder for cloud client config fetch; no infra implementation."""
    return {
        "client_id": client_id,
        "source": "cloud",
        "status": "remote config fetch not implemented",
    }


def _ensure_final_response_contract(response: Any) -> FinalInsightResponse:
    if not isinstance(response, FinalInsightResponse):
        raise RuntimeError("Invalid response contract.")
    try:
        serialized = response.model_dump_json()
        payload = json.loads(serialized)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Invalid response contract.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid response contract.")
    if set(payload.keys()) != _FINAL_RESPONSE_KEYS:
        raise RuntimeError("Invalid response contract.")
    try:
        FinalInsightResponse.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Invalid response contract.") from exc
    return response


def _api_error_message(resp: requests.Response) -> str:
    """Extract a useful error message from non-2xx backend responses."""
    try:
        payload = resp.json()
    except ValueError:
        text = (resp.text or "").strip()
        return text or f"HTTP {resp.status_code}"

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, dict):
            code = str(detail.get("code") or "").strip()
            err_type = str(detail.get("error_type") or "").strip()
            message = str(detail.get("message") or "").strip()
            if code and message:
                return f"{code}: {message}"
            if err_type and message:
                return f"{err_type}: {message}"
            if message:
                return message
            return json.dumps(detail)
        if detail is not None:
            return str(detail)
        return json.dumps(payload)
    return str(payload)


def run_pipeline(
    data: bytes | None,
    prompt: str,
    filename: str | None = None,
    client_id: str | None = None,
    business_type: str | None = None,
    multi_entity_behavior: str | None = None,
    client_config: Optional[dict[str, Any]] = None,
    model: str = "default",
) -> FinalInsightResponse:
    """Call the backend /analyze endpoint via HTTP."""
    try:
        form_data: dict[str, str] = {"prompt": prompt}
        if client_id:
            form_data["client_id"] = client_id
        if business_type:
            form_data["business_type"] = business_type
        if multi_entity_behavior:
            form_data["multi_entity_behavior"] = multi_entity_behavior
        if model and model != "default":
            form_data["model"] = model

        files = {}
        if data is not None:
            files["file"] = (filename or "upload.csv", io.BytesIO(data), "text/csv")

        resp = requests.post(
            f"{API_BASE_URL}/analyze",
            data=form_data,
            files=files if files else None,
            timeout=120,
        )
        if resp.status_code >= 400:
            reason = _api_error_message(resp)
            return _ensure_final_response_contract(
                FinalInsightResponse.failure(f"API error ({resp.status_code}): {reason}")
            )
        payload = resp.json()
        return _ensure_final_response_contract(FinalInsightResponse(**payload))
    except requests.RequestException as exc:
        logger.exception("API call to /analyze failed")
        return _ensure_final_response_contract(
            FinalInsightResponse.failure(f"API error: {exc}")
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline execution failed client_id=%r", client_id)
        return _ensure_final_response_contract(FinalInsightResponse.failure(str(exc)))


def fetch_report_payload(
    *,
    entity_name: str,
    prompt: str,
    business_type: str | None,
) -> dict[str, Any] | None:
    """Fetch backend-generated report payload for charts and report download."""
    try:
        params: dict[str, str] = {
            "entity_name": entity_name,
            "prompt": prompt,
            "format": "json",
        }
        if business_type:
            params["business_type"] = business_type
        resp = requests.get(
            f"{API_BASE_URL}/export/report",
            params=params,
            timeout=120,
        )
        if resp.status_code >= 400:
            logger.warning("Report payload fetch failed: %s", _api_error_message(resp))
            return None
        payload = resp.json()
        return payload if isinstance(payload, dict) else None
    except requests.RequestException:
        logger.exception("Report payload request failed")
        return None


def fetch_export_bytes(
    *,
    dataset: str,
    output_format: str,
    entity_name: str | None,
) -> bytes | None:
    """Download backend export bytes for CSV/PowerBI actions."""
    try:
        params: dict[str, str] = {
            "dataset": dataset,
            "format": output_format,
        }
        if entity_name:
            params["entity_name"] = entity_name
        resp = requests.get(
            f"{API_BASE_URL}/export/powerbi",
            params=params,
            timeout=120,
        )
        if resp.status_code >= 400:
            logger.warning("Export fetch failed: %s", _api_error_message(resp))
            return None
        return resp.content
    except requests.RequestException:
        logger.exception("Export request failed")
        return None


def fetch_export_json(
    *,
    dataset: str,
    entity_name: str | None,
    limit: int = 2000,
) -> dict[str, Any] | None:
    """Fetch backend export JSON payload for chart rendering."""
    try:
        params: dict[str, str | int] = {
            "dataset": dataset,
            "format": "json",
            "limit": limit,
        }
        if entity_name:
            params["entity_name"] = entity_name
        resp = requests.get(
            f"{API_BASE_URL}/export/powerbi",
            params=params,
            timeout=120,
        )
        if resp.status_code >= 400:
            logger.warning("Export JSON fetch failed: %s", _api_error_message(resp))
            return None
        payload = resp.json()
        return payload if isinstance(payload, dict) else None
    except requests.RequestException:
        logger.exception("Export JSON request failed")
        return None


def fetch_report_markdown(
    *,
    entity_name: str,
    prompt: str,
    business_type: str | None,
) -> bytes | None:
    """Download backend-formatted report markdown."""
    try:
        params: dict[str, str] = {
            "entity_name": entity_name,
            "prompt": prompt,
            "format": "md",
        }
        if business_type:
            params["business_type"] = business_type
        resp = requests.get(
            f"{API_BASE_URL}/export/report",
            params=params,
            timeout=120,
        )
        if resp.status_code >= 400:
            logger.warning("Report markdown fetch failed: %s", _api_error_message(resp))
            return None
        return resp.content
    except requests.RequestException:
        logger.exception("Report markdown request failed")
        return None


def _first_available(mapping: dict[str, Any], keys: list[str]) -> Any:
    """Return the first available backend-provided value by key."""
    for key in keys:
        if key in mapping and mapping.get(key) is not None:
            return mapping.get(key)
    return None


def _insight_record(output: InsightOutput) -> dict[str, Any]:
    """Return normalized base fields from InsightOutput."""
    return {
        "insight": output.insight,
        "evidence": output.evidence,
        "impact": output.impact,
        "recommended_action": output.recommended_action,
        "priority": output.priority,
        "confidence_score": output.confidence_score,
        "pipeline_status": output.pipeline_status,
    }


def _extract_client_id(raw_state: dict[str, Any], fallback_client_id: Any = None) -> Any:
    """Return client_id if backend provided one."""
    direct = _first_available(raw_state, ["client_id"])
    if direct is not None:
        return direct

    kpi_payload = (
        raw_state.get("saas_kpi_data")
        or raw_state.get("ecommerce_kpi_data")
        or raw_state.get("agency_kpi_data")
    )
    if isinstance(kpi_payload, dict):
        payload_client_id = kpi_payload.get("client_id")
        if payload_client_id is not None:
            return payload_client_id

        records = kpi_payload.get("records")
        if isinstance(records, list) and records:
            latest = records[-1]
            if isinstance(latest, dict):
                return latest.get("client_id")

    return fallback_client_id


def _build_powerbi_record(
    output: InsightOutput,
    raw_state: dict[str, Any],
    fallback_client_id: Any = None,
) -> dict[str, Any]:
    """Build one PowerBI-ready structured record."""
    record = _insight_record(output)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    record["client_id"] = _extract_client_id(raw_state, fallback_client_id=fallback_client_id)
    return record


def _build_run_signature(
    *,
    prompt: str,
    mode: str,
    model: str,
    client_id: Optional[str],
    business_type: Optional[str],
    multi_entity_behavior: Optional[str],
    config_enabled: bool,
    upload_hash: Optional[str],
) -> str:
    """Build deterministic signature used to skip unnecessary reruns."""
    payload = {
        "prompt": prompt.strip(),
        "mode": mode,
        "model": model,
        "client_id": client_id or "",
        "business_type": business_type or "",
        "multi_entity_behavior": multi_entity_behavior or "",
        "config_enabled": config_enabled,
        "upload_hash": upload_hash or "",
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None
if "uploaded_hash" not in st.session_state:
    st.session_state.uploaded_hash = None
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "pipeline_error" not in st.session_state:
    st.session_state.pipeline_error = None
if "execution_time_s" not in st.session_state:
    st.session_state.execution_time_s = None
if "last_run_signature" not in st.session_state:
    st.session_state.last_run_signature = None
if "used_cached_run" not in st.session_state:
    st.session_state.used_cached_run = False
if "last_client_id" not in st.session_state:
    st.session_state.last_client_id = None
if "report_payload" not in st.session_state:
    st.session_state.report_payload = None
if "export_csv_bytes" not in st.session_state:
    st.session_state.export_csv_bytes = None
if "export_powerbi_bytes" not in st.session_state:
    st.session_state.export_powerbi_bytes = None
if "report_markdown_bytes" not in st.session_state:
    st.session_state.report_markdown_bytes = None
if "export_kpi_json" not in st.session_state:
    st.session_state.export_kpi_json = None


with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", options=["LOCAL", "CLOUD"], horizontal=True)
    client_options = _discover_clients()
    selected_client = st.selectbox("Client", options=client_options, index=0)
    manual_client_id = st.text_input(
        "Entity / Client ID Override",
        value="",
        help="Used as client_id for analysis. Required when no CSV is uploaded.",
    )
    business_type_option = st.selectbox(
        "Business Type",
        options=_BUSINESS_TYPE_OPTIONS,
        index=0,
        help="Optional override; leave 'auto' to let backend infer from prompt.",
    )
    multi_entity_behavior_option = st.selectbox(
        "Multi-Entity CSV Handling",
        options=_MULTI_ENTITY_OPTIONS,
        index=0,
        help="Used when uploaded CSV contains multiple entity_name values.",
    )
    load_client_config = st.checkbox("Load Client Configuration", value=False)
    model = st.selectbox(
        "Model",
        options=["default", "gpt-5.4"],
        index=0,
    )

    client_config: Optional[dict[str, Any]] = None
    client_config_note: str | None = None
    if load_client_config:
        if mode == "LOCAL":
            client_config = _load_local_client_config(selected_client)
            if selected_client != "default" and client_config is None:
                client_config_note = "Local client config not found."
            elif client_config is not None:
                client_config_note = "Loaded local client config."
        else:
            client_config = _fetch_remote_client_config_placeholder(selected_client)
            client_config_note = "Using cloud config placeholder."

    if client_config_note:
        st.caption(client_config_note)

    run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)
    if st.button("Clear", use_container_width=True):
        st.session_state.pipeline_result = None
        st.session_state.pipeline_error = None
        st.session_state.execution_time_s = None
        st.session_state.last_run_signature = None
        st.session_state.used_cached_run = False
        st.session_state.last_client_id = None
        st.session_state.report_payload = None
        st.session_state.export_csv_bytes = None
        st.session_state.export_powerbi_bytes = None
        st.session_state.report_markdown_bytes = None
        st.session_state.export_kpi_json = None
        st.rerun()


st.title("InsightAgent")

st.subheader("Section 1: Historical Data Upload")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
preview_rows = st.slider("Preview rows", min_value=5, max_value=100, value=20, step=5)
if uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()
    st.session_state.uploaded_bytes = uploaded_bytes
    st.session_state.uploaded_name = uploaded_file.name
    st.session_state.uploaded_hash = hashlib.sha256(uploaded_bytes).hexdigest()

if st.session_state.uploaded_bytes is not None:
    try:
        preview_df, column_list = _load_csv_preview(
            st.session_state.uploaded_bytes,
            preview_rows,
        )
        st.dataframe(preview_df, use_container_width=True)
        st.caption(f"Showing first {len(preview_df)} row(s).")
        st.write("Columns:")
        st.code(", ".join(column_list) if column_list else "-")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load CSV preview: {exc}")
else:
    st.info("Upload a CSV file to preview data.")

st.subheader("Section 2: Strategic Prompt Input")
prompt = st.text_area(
    "Business Prompt",
    placeholder="Enter a strategic business prompt.",
    height=140,
)

if run_clicked:
    if not prompt.strip():
        st.session_state.pipeline_result = None
        st.session_state.pipeline_error = "Prompt is required before running analysis."
        st.session_state.used_cached_run = False
    else:
        manual = manual_client_id.strip()
        selected = None if selected_client == "default" else selected_client
        chosen_client_id = manual or selected
        chosen_business_type = (
            None if business_type_option == "auto" else business_type_option
        )
        chosen_multi_entity_behavior = (
            None if multi_entity_behavior_option == "auto" else multi_entity_behavior_option
        )

        can_run = True
        if st.session_state.uploaded_bytes is None and not chosen_client_id:
            st.session_state.pipeline_result = None
            st.session_state.pipeline_error = (
                "Entity / client_id is required when no CSV is uploaded."
            )
            st.session_state.used_cached_run = False
            st.session_state.last_client_id = None
            can_run = False

        if can_run:
            run_signature = _build_run_signature(
                prompt=prompt,
                mode=mode,
                model=model,
                client_id=chosen_client_id,
                business_type=chosen_business_type,
                multi_entity_behavior=chosen_multi_entity_behavior,
                config_enabled=load_client_config,
                upload_hash=st.session_state.uploaded_hash,
            )

            if (
                st.session_state.pipeline_result is not None
                and st.session_state.last_run_signature == run_signature
            ):
                st.session_state.pipeline_error = None
                st.session_state.used_cached_run = True
            else:
                st.session_state.used_cached_run = False

            if not st.session_state.used_cached_run:
                with st.spinner("Running analysis pipeline..."):
                    started = time.perf_counter()
                    try:
                        st.session_state.pipeline_result = run_pipeline(
                            data=st.session_state.uploaded_bytes,
                            prompt=prompt.strip(),
                            filename=st.session_state.uploaded_name,
                            client_id=chosen_client_id,
                            business_type=chosen_business_type,
                            multi_entity_behavior=chosen_multi_entity_behavior,
                            client_config=client_config,
                            model=model,
                        )
                        st.session_state.pipeline_error = None
                        st.session_state.execution_time_s = time.perf_counter() - started
                        st.session_state.last_run_signature = run_signature
                        st.session_state.last_client_id = chosen_client_id
                        st.session_state.report_payload = (
                            fetch_report_payload(
                                entity_name=chosen_client_id or "",
                                prompt=prompt.strip(),
                                business_type=chosen_business_type,
                            )
                            if chosen_client_id
                            else None
                        )
                        st.session_state.export_csv_bytes = fetch_export_bytes(
                            dataset="records",
                            output_format="csv",
                            entity_name=chosen_client_id,
                        )
                        st.session_state.export_powerbi_bytes = fetch_export_bytes(
                            dataset="kpis",
                            output_format="csv",
                            entity_name=chosen_client_id,
                        )
                        st.session_state.export_kpi_json = fetch_export_json(
                            dataset="kpis",
                            entity_name=chosen_client_id,
                        )
                        st.session_state.report_markdown_bytes = (
                            fetch_report_markdown(
                                entity_name=chosen_client_id or "",
                                prompt=prompt.strip(),
                                business_type=chosen_business_type,
                            )
                            if chosen_client_id
                            else None
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.session_state.pipeline_result = None
                        st.session_state.pipeline_error = f"Pipeline error: {exc}"
                        st.session_state.execution_time_s = None
                        st.session_state.last_client_id = None
                        st.session_state.report_payload = None
                        st.session_state.export_csv_bytes = None
                        st.session_state.export_powerbi_bytes = None
                        st.session_state.report_markdown_bytes = None
                        st.session_state.export_kpi_json = None


st.subheader("Section 3: Execution Results")
if st.session_state.pipeline_error:
    st.error(st.session_state.pipeline_error)
elif st.session_state.pipeline_result is None:
    st.info("Run analysis to view results.")
else:
    output: FinalInsightResponse = st.session_state.pipeline_result
    selected_client_id: str | None = st.session_state.last_client_id
    report_payload = (
        st.session_state.report_payload
        if isinstance(st.session_state.report_payload, dict)
        else {}
    )
    derived_signals = (
        report_payload.get("derived_signals")
        if isinstance(report_payload.get("derived_signals"), dict)
        else {}
    )

    if st.session_state.execution_time_s is not None:
        st.caption(f"Execution time: {st.session_state.execution_time_s:.2f}s")
    if st.session_state.used_cached_run:
        st.caption("Using previous result (inputs unchanged).")

    st.json(output.model_dump())
    st.markdown(f"**Insight:** {output.insight}")
    st.markdown(f"**Evidence:** {output.evidence}")
    st.markdown(f"**Impact:** {output.impact}")
    st.markdown(f"**Recommended Action:** {output.recommended_action}")
    st.markdown(f"**Priority:** {output.priority}")
    st.markdown(f"**Confidence Score:** {output.confidence_score}")
    st.markdown(f"**Pipeline Status:** {output.pipeline_status}")

    if output.diagnostics is not None:
        st.markdown("**Ingestion / Signal Diagnostics**")
        st.json(output.diagnostics.model_dump())

    kpi_export = (
        st.session_state.export_kpi_json
        if isinstance(st.session_state.export_kpi_json, dict)
        else {}
    )
    kpi_rows = kpi_export.get("data") if isinstance(kpi_export.get("data"), list) else []
    if kpi_rows:
        kpi_df = pd.DataFrame(kpi_rows)
        required_cols = {"period_end", "metric_name", "metric_value"}
        if required_cols.issubset(set(kpi_df.columns)):
            kpi_df["period_end"] = pd.to_datetime(kpi_df["period_end"], errors="coerce")
            kpi_df = kpi_df.dropna(subset=["period_end"])
            if not kpi_df.empty:
                chart_df = kpi_df.pivot_table(
                    index="period_end",
                    columns="metric_name",
                    values="metric_value",
                    aggfunc="last",
                ).sort_index()
                st.markdown("**Time-Series Chart**")
                st.line_chart(chart_df, use_container_width=True)

    role_payload = derived_signals.get("role_contribution")
    if isinstance(role_payload, dict):
        contributors = role_payload.get("top_contributors")
        if isinstance(contributors, list) and contributors:
            role_df = pd.DataFrame(contributors)
            if {"name", "contribution_value"}.issubset(set(role_df.columns)):
                st.markdown("**Role Contribution Chart**")
                st.bar_chart(
                    role_df.set_index("name")["contribution_value"],
                    use_container_width=True,
                )

    risk_payload = derived_signals.get("risk")
    if isinstance(risk_payload, dict):
        risk_score = risk_payload.get("risk_score")
        try:
            risk_score_value = float(risk_score)
        except (TypeError, ValueError):
            risk_score_value = None
        if risk_score_value is not None:
            st.markdown("**Risk Gauge**")
            st.progress(min(100, max(0, int(round(risk_score_value)))) / 100.0)
            st.caption(f"Risk score: {risk_score_value:.2f}")

    multivariate_payload = derived_signals.get("multivariate_scenario")
    if isinstance(multivariate_payload, dict):
        scenario_simulation = multivariate_payload.get("scenario_simulation")
        scenarios = (
            scenario_simulation.get("scenarios")
            if isinstance(scenario_simulation, dict)
            else None
        )
        if isinstance(scenarios, dict):
            rows: list[dict[str, Any]] = []
            for name, scenario in scenarios.items():
                if not isinstance(scenario, dict):
                    continue
                rows.append(
                    {
                        "scenario": name,
                        "projected_value": scenario.get("projected_value"),
                        "projected_growth": scenario.get("projected_growth"),
                    }
                )
            if rows:
                scenario_df = pd.DataFrame(rows).set_index("scenario")
                st.markdown("**Scenario Comparison**")
                st.bar_chart(
                    scenario_df[["projected_value", "projected_growth"]],
                    use_container_width=True,
                )

    json_bytes = json.dumps(output.model_dump(), indent=2).encode("utf-8")
    csv_bytes = (
        st.session_state.export_csv_bytes
        if isinstance(st.session_state.export_csv_bytes, (bytes, bytearray))
        else None
    )
    powerbi_bytes = (
        st.session_state.export_powerbi_bytes
        if isinstance(st.session_state.export_powerbi_bytes, (bytes, bytearray))
        else None
    )
    report_md_bytes = (
        st.session_state.report_markdown_bytes
        if isinstance(st.session_state.report_markdown_bytes, (bytes, bytearray))
        else None
    )

    dcol1, dcol2, dcol3, dcol4 = st.columns(4)
    with dcol1:
        st.download_button(
            label="Download JSON",
            data=json_bytes,
            file_name="insight_output.json",
            mime="application/json",
            use_container_width=True,
        )
    with dcol2:
        st.download_button(
            label="Download CSV",
            data=csv_bytes or b"",
            file_name="insight_records.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=csv_bytes is None,
        )
    with dcol3:
        st.download_button(
            label="Download PowerBI Dataset",
            data=powerbi_bytes or b"",
            file_name="powerbi_dataset.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=powerbi_bytes is None,
        )
    with dcol4:
        st.download_button(
            label="Download Report",
            data=report_md_bytes or b"",
            file_name="insight_report.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=report_md_bytes is None,
        )

    with st.expander("Show Internal Analysis"):
        st.json(
            {
                "final_insight_response": output.model_dump(),
                "report_payload": report_payload,
            }
        )

    # --- Intelligence Dashboard (React embed) ---
    st.subheader("Section 4: Intelligence Dashboard")
    entity_for_dashboard = (
        st.session_state.last_client_id or manual_client_id.strip() or ""
    )
    btype_for_dashboard = (
        business_type_option if business_type_option != "auto" else "saas"
    )
    if entity_for_dashboard:
        from urllib.parse import urlencode

        embed_params = urlencode({
            "embed": "1",
            "entity_name": entity_for_dashboard,
            "business_type": btype_for_dashboard,
        })
        embed_url = f"{API_BASE_URL}/dashboard?{embed_params}"
        components.iframe(embed_url, height=900, scrolling=True)
    else:
        st.info(
            "Intelligence Dashboard requires an entity name. "
            "Provide one via the sidebar and re-run analysis."
        )
