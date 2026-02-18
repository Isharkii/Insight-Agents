"""Minimal Streamlit frontend for the modular Insight Agent."""

from __future__ import annotations

import io
import json
import os
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
from fastapi import UploadFile
from pydantic import ValidationError

from llm_synthesis.schema import InsightOutput

st.set_page_config(page_title="InsightAgent", page_icon="IA", layout="wide")


@st.cache_resource(show_spinner=False)
def _load_backend_handles():
    """Load backend orchestrators lazily to keep startup lightweight."""
    from agent.graph import insight_graph  # noqa: PLC0415
    from agent.nodes.intent import intent_node  # noqa: PLC0415
    from app.services.csv_ingestion_service import get_csv_ingestion_service  # noqa: PLC0415
    from db.session import SessionLocal  # noqa: PLC0415

    return {
        "graph": insight_graph,
        "intent_node": intent_node,
        "csv_service": get_csv_ingestion_service(),
        "session_factory": SessionLocal,
    }


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


def _extract_output(state: dict[str, Any]) -> InsightOutput:
    """Extract and validate final structured output from pipeline state."""
    response = state.get("final_response")
    if not isinstance(response, str):
        raise ValueError("Pipeline response is missing final_response.")

    try:
        payload = json.loads(response)
        return InsightOutput.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise ValueError(f"Pipeline output is not valid InsightOutput: {exc}") from exc


def run_pipeline(
    data: bytes | None,
    prompt: str,
    filename: str | None = None,
    client_id: str | None = None,
    client_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Thin frontend adapter that delegates all processing to backend layers."""
    handles = _load_backend_handles()
    graph = handles["graph"]
    intent_node = handles["intent_node"]
    csv_service = handles["csv_service"]
    session_factory = handles["session_factory"]

    seed_state = intent_node({"user_query": prompt})
    business_type = seed_state.get("business_type")
    entity_name = seed_state.get("entity_name")

    allowed_business_types = {"saas", "ecommerce", "agency"}
    ingest_business_type = business_type if business_type in allowed_business_types else None
    ingest_entity_name = entity_name or client_id or None

    ingestion_summary: dict[str, Any] | None = None
    if data is not None:
        upload = UploadFile(filename=filename or "upload.csv", file=io.BytesIO(data))
        try:
            with session_factory() as db:
                summary = csv_service.ingest_csv(
                    upload_file=upload,
                    db=db,
                    client_name=ingest_entity_name,
                    business_type=ingest_business_type,
                )
            ingestion_summary = {
                "rows_processed": summary.rows_processed,
                "rows_failed": summary.rows_failed,
                "validation_error_count": len(summary.validation_errors),
            }
        finally:
            upload.file.close()

    invoke_state: dict[str, Any] = {"user_query": prompt}
    if ingest_business_type:
        invoke_state["business_type"] = ingest_business_type
    if ingest_entity_name:
        invoke_state["entity_name"] = ingest_entity_name

    state = graph.invoke(invoke_state)
    output = _extract_output(state)

    return {
        "output": output,
        "output_json": output.model_dump(),
        "raw_state": state,
        "ingestion_summary": ingestion_summary,
        "client_id": client_id,
        "client_config": client_config,
    }


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
    config_enabled: bool,
    upload_hash: Optional[str],
) -> str:
    """Build deterministic signature used to skip unnecessary reruns."""
    payload = {
        "prompt": prompt.strip(),
        "mode": mode,
        "model": model,
        "client_id": client_id or "",
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


with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", options=["LOCAL", "CLOUD"], horizontal=True)
    client_options = _discover_clients()
    selected_client = st.selectbox("Client", options=client_options, index=0)
    load_client_config = st.checkbox("Load Client Configuration", value=False)
    model = st.selectbox("Model", options=["default", "gpt-4o-mini", "gpt-4o"], index=0)

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
        chosen_client_id = None if selected_client == "default" else selected_client
        run_signature = _build_run_signature(
            prompt=prompt,
            mode=mode,
            model=model,
            client_id=chosen_client_id,
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

        os.environ["LLM_ADAPTER"] = "mock" if mode == "LOCAL" else "openai"
        if model != "default":
            os.environ["LLM_MODEL"] = model

        if not st.session_state.used_cached_run:
            with st.spinner("Running analysis pipeline..."):
                started = time.perf_counter()
                try:
                    st.session_state.pipeline_result = run_pipeline(
                        data=st.session_state.uploaded_bytes,
                        prompt=prompt.strip(),
                        filename=st.session_state.uploaded_name,
                        client_id=chosen_client_id,
                        client_config=client_config,
                    )
                    st.session_state.pipeline_error = None
                    st.session_state.execution_time_s = time.perf_counter() - started
                    st.session_state.last_run_signature = run_signature
                except Exception as exc:  # noqa: BLE001
                    st.session_state.pipeline_result = None
                    st.session_state.pipeline_error = f"Pipeline error: {exc}"
                    st.session_state.execution_time_s = None


st.subheader("Section 3: Execution Results")
if st.session_state.pipeline_error:
    st.error(st.session_state.pipeline_error)
elif st.session_state.pipeline_result is None:
    st.info("Run analysis to view results.")
else:
    result = st.session_state.pipeline_result
    output: InsightOutput = result["output"]
    raw_state: dict[str, Any] = result.get("raw_state", {})
    selected_client_id = result.get("client_id")
    loaded_client_config = result.get("client_config")

    if result.get("ingestion_summary") is not None:
        st.caption(f"Ingestion: {result['ingestion_summary']}")
    if selected_client_id is not None:
        st.caption(f"Client: {selected_client_id}")
    if loaded_client_config is not None:
        st.caption("Client configuration loaded.")
    if st.session_state.execution_time_s is not None:
        st.caption(f"Execution time: {st.session_state.execution_time_s:.2f}s")
    if st.session_state.used_cached_run:
        st.caption("Using previous result (inputs unchanged).")

    st.json(result["output_json"])
    st.markdown(f"**Insight:** {output.insight}")
    st.markdown(f"**Evidence:** {output.evidence}")
    st.markdown(f"**Impact:** {output.impact}")
    st.markdown(f"**Recommended Action:** {output.recommended_action}")
    st.markdown(f"**Priority:** {output.priority}")
    st.markdown(f"**Confidence Score:** {output.confidence_score}")

    powerbi_record = _build_powerbi_record(
        output,
        raw_state,
        fallback_client_id=selected_client_id,
    )
    summary_record = _insight_record(output)

    json_bytes = json.dumps(powerbi_record, indent=2).encode("utf-8")

    csv_buffer = io.StringIO()
    pd.DataFrame([summary_record]).to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    powerbi_columns = [
        "insight",
        "evidence",
        "impact",
        "recommended_action",
        "priority",
        "confidence_score",
        "timestamp",
        "client_id",
    ]
    powerbi_buffer = io.StringIO()
    pd.DataFrame([powerbi_record], columns=powerbi_columns).to_csv(
        powerbi_buffer,
        index=False,
    )
    powerbi_bytes = powerbi_buffer.getvalue().encode("utf-8")

    dcol1, dcol2, dcol3 = st.columns(3)
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
            data=csv_bytes,
            file_name="insight_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dcol3:
        st.download_button(
            label="Download PowerBI Dataset",
            data=powerbi_bytes,
            file_name="powerbi_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("Show Internal Analysis"):
        st.markdown("**Extracted Data Features**")

        kpi_payload = (
            raw_state.get("saas_kpi_data")
            or raw_state.get("ecommerce_kpi_data")
            or raw_state.get("agency_kpi_data")
        )
        kpi_metrics = None
        if isinstance(kpi_payload, dict):
            records = kpi_payload.get("records")
            if isinstance(records, list) and records:
                latest = records[-1]
                if isinstance(latest, dict):
                    kpi_metrics = latest.get("computed_kpis")
            if kpi_metrics is None:
                kpi_metrics = kpi_payload.get("computed_kpis")

        anomalies = _first_available(
            raw_state,
            ["detected_anomalies", "anomalies"],
        )
        trends = _first_available(
            raw_state,
            ["trends"],
        )
        if trends is None and isinstance(raw_state.get("forecast_data"), dict):
            trends = _first_available(raw_state["forecast_data"], ["trend", "trends"])

        st.markdown("KPI metrics:")
        st.json(kpi_metrics if kpi_metrics is not None else "Not returned by backend")
        st.markdown("Detected anomalies:")
        st.json(anomalies if anomalies is not None else "Not returned by backend")
        st.markdown("Trends:")
        st.json(trends if trends is not None else "Not returned by backend")

        st.markdown("**Extracted Prompt Features**")
        intent = _first_available(
            raw_state,
            ["intent", "detected_intent", "business_type"],
        )
        risk_focus = _first_available(
            raw_state,
            ["risk_focus"],
        )
        if risk_focus is None and isinstance(raw_state.get("prioritization"), dict):
            risk_focus = _first_available(raw_state["prioritization"], ["risk_focus"])
        opportunity_focus = _first_available(
            raw_state,
            ["opportunity_focus"],
        )
        if opportunity_focus is None and isinstance(raw_state.get("prioritization"), dict):
            opportunity_focus = _first_available(
                raw_state["prioritization"],
                ["opportunity_focus", "recommended_focus"],
            )

        st.markdown("Intent:")
        st.json(intent if intent is not None else "Not returned by backend")
        st.markdown("Risk focus:")
        st.json(risk_focus if risk_focus is not None else "Not returned by backend")
        st.markdown("Opportunity focus:")
        st.json(opportunity_focus if opportunity_focus is not None else "Not returned by backend")

        if loaded_client_config is not None:
            st.markdown("**Client Configuration**")
            st.json(loaded_client_config)
