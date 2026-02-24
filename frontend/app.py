"""Streamlit frontend for InsightAgent.

Alternative local UI that invokes the LangGraph directly.
"""

from __future__ import annotations

import io
import json
from typing import Any, Optional

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from llm_synthesis.schema import FinalInsightResponse

st.set_page_config(
    page_title="InsightAgent",
    page_icon="IA",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def _load_graph():
    from agent.graph import insight_graph  # noqa: PLC0415

    return insight_graph


_STATE_DEFAULTS: dict[str, Any] = {
    "result": None,
    "uploaded_df": None,
}
for _key, _val in _STATE_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


with st.sidebar:
    st.title("InsightAgent")
    st.caption("Modular AI Insight Engine")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload data (CSV)",
        type=["csv"],
        help="Optional: preview your dataset before running analysis.",
    )
    if uploaded_file:
        df = pd.read_csv(io.BytesIO(uploaded_file.read()))
        st.session_state.uploaded_df = df
        st.success(f"Loaded {len(df):,} rows x {len(df.columns)} columns")

    st.divider()
    user_query = st.text_area(
        "Query",
        placeholder="e.g. Analyse SaaS KPIs for Acme Cloud",
        height=100,
    )
    run = st.button("Run Analysis", type="primary", use_container_width=True)

    if st.session_state.result and st.button("Clear", use_container_width=True):
        st.session_state.result = None
        st.session_state.uploaded_df = None
        st.rerun()


if run and user_query.strip():
    with st.spinner("Running insight pipeline..."):
        try:
            graph = _load_graph()
            state = graph.invoke({"user_query": user_query.strip()})
            st.session_state.result = state
        except Exception as exc:  # noqa: BLE001
            st.error(f"Pipeline error: {exc}")
elif run:
    st.sidebar.warning("Please enter a query before running.")


def _unwrap_node_payload(value: Any) -> dict[str, Any]:
    """Support both legacy raw payloads and new status/payload envelopes."""
    if isinstance(value, dict) and value.get("status") in {"success", "skipped", "failed"}:
        payload = value.get("payload")
        return payload if isinstance(payload, dict) else {}
    return value if isinstance(value, dict) else {}


def _render_upload_preview() -> None:
    df: Optional[pd.DataFrame] = st.session_state.uploaded_df
    if df is None:
        st.info("Upload a CSV file in the sidebar to preview it here.")
        return

    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    cols = st.columns(3)
    cols[0].metric("Rows", f"{len(df):,}")
    cols[1].metric("Columns", len(df.columns))
    cols[2].metric("Nulls", int(df.isnull().sum().sum()))


def _render_kpi_dashboard(result: dict[str, Any]) -> None:
    raw_kpi: Optional[dict[str, Any]] = (
        result.get("saas_kpi_data")
        or result.get("ecommerce_kpi_data")
        or result.get("agency_kpi_data")
    )
    if not raw_kpi:
        st.info("No KPI data available for this run.")
        return

    kpi = _unwrap_node_payload(raw_kpi)
    records = kpi.get("records", [])
    if not isinstance(records, list) or not records:
        st.warning("KPI records are empty.")
        st.json(raw_kpi)
        return

    latest = records[-1].get("computed_kpis", {}) if isinstance(records[-1], dict) else {}
    if not isinstance(latest, dict) or not latest:
        st.json(raw_kpi)
        return

    st.subheader("KPI Metrics")
    cols = st.columns(min(len(latest), 4))
    for idx, (name, val) in enumerate(latest.items()):
        if isinstance(val, dict) and "value" in val:
            val = val.get("value")
        display = f"{val:.2f}" if isinstance(val, float) else str(val)
        cols[idx % 4].metric(name.replace("_", " ").title(), display)

    with st.expander("Raw KPI payload"):
        st.json(raw_kpi)


def _render_risk(result: dict[str, Any]) -> None:
    raw_risk: Optional[dict[str, Any]] = result.get("risk_data")
    raw_root: Optional[dict[str, Any]] = result.get("root_cause")

    if not raw_risk:
        st.info("No risk data available.")
        return

    risk = _unwrap_node_payload(raw_risk)
    root = _unwrap_node_payload(raw_root)

    score = int(risk.get("risk_score", 0) or 0)
    level = str(risk.get("risk_level", "unknown")).upper()

    level_color = {
        "LOW": "green",
        "MODERATE": "orange",
        "HIGH": "orangered",
        "CRITICAL": "red",
    }

    cols = st.columns([1, 3])
    with cols[0]:
        st.metric("Risk Score", score, help="0 = no risk, 100 = critical")
        st.markdown(f"**Level:** :{level_color.get(level, 'gray')}[{level}]")
    with cols[1]:
        st.progress(max(0.0, min(1.0, score / 100.0)), text=f"{score} / 100")

    if root:
        st.subheader("Root Cause Analysis")
        causes = root.get("root_causes") or root.get("primary_issue") or "-"
        evidence = root.get("evidence", [])
        if isinstance(causes, list):
            for c in causes:
                st.markdown(f"- {c}")
        else:
            st.markdown(f"**Primary issue:** {causes}")

        if evidence:
            with st.expander("Evidence"):
                for e in evidence:
                    st.markdown(f"- {e}")

        if root.get("recommended_action"):
            st.info(f"**Suggested action:** {root['recommended_action']}")

    with st.expander("Raw risk and root-cause payload"):
        st.json({"risk_data": raw_risk, "root_cause": raw_root})


def _render_segments(result: dict[str, Any]) -> None:
    raw_seg: Optional[dict[str, Any]] = result.get("segmentation")
    if not raw_seg:
        st.info("No role analytics or segmentation data found.")
        return

    seg = _unwrap_node_payload(raw_seg)

    role_scoring = seg.get("role_scoring")
    if isinstance(role_scoring, dict) and role_scoring:
        st.subheader("Role Analytics")
        rows: list[dict[str, Any]] = []
        for role, payload in role_scoring.items():
            if not isinstance(payload, dict):
                continue
            rows.append(
                {
                    "Role": role,
                    "Performance Score": payload.get("performance_score"),
                    "Classification": payload.get("classification"),
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        with st.expander("Raw role analytics payload"):
            st.json(raw_seg)
        return

    if not seg.get("found"):
        st.info("No segmentation data found for this entity.")
        return

    segment_data: dict[str, Any] = seg.get("segment_data", {})
    if not segment_data:
        st.json(raw_seg)
        return

    st.subheader(f"Segments - {seg.get('n_clusters', '?')} clusters")
    for label, profile in segment_data.items():
        with st.expander(label.replace("_", " ").title()):
            if isinstance(profile, dict):
                rows = [{"Metric": k, "Value": v} for k, v in profile.items()]
                st.table(pd.DataFrame(rows))
            else:
                st.write(profile)


def _render_insights(result: dict[str, Any]) -> None:
    response: Optional[str] = result.get("final_response")

    try:
        if not response:
            raise ValueError("Pipeline response is missing final_response.")
        payload = json.loads(response)
        parsed = FinalInsightResponse.model_validate(payload)
    except (json.JSONDecodeError, TypeError, ValidationError, ValueError) as exc:
        parsed = FinalInsightResponse.failure(reason=str(exc))

    st.subheader("Insight")
    st.write(parsed.insight)

    st.subheader("Evidence")
    st.write(parsed.evidence)

    st.subheader("Impact")
    st.warning(parsed.impact)

    st.subheader("Recommended Action")
    st.markdown(f"- {parsed.recommended_action}")

    st.subheader("Priority")
    st.write(parsed.priority)

    st.subheader("Pipeline Status")
    st.write(parsed.pipeline_status)

    conf = float(parsed.confidence_score)
    st.progress(max(0.0, min(1.0, conf)), text=f"Confidence: {conf:.0%}")

    with st.expander("Raw agent state"):
        safe = {k: v for k, v in result.items() if k != "final_response"}
        st.json(safe)


result: Optional[dict[str, Any]] = st.session_state.result


(tab_upload, tab_kpi, tab_risk, tab_segments, tab_insights) = st.tabs(
    ["Upload", "KPI Dashboard", "Risk", "Segments", "Insights"]
)

with tab_upload:
    _render_upload_preview()

with tab_kpi:
    if result:
        _render_kpi_dashboard(result)
    else:
        st.info("Run an analysis from the sidebar to see KPI data.")

with tab_risk:
    if result:
        _render_risk(result)
    else:
        st.info("Run an analysis from the sidebar to see risk data.")

with tab_segments:
    if result:
        _render_segments(result)
    else:
        st.info("Run an analysis from the sidebar to see role analytics or segmentation data.")

with tab_insights:
    if result:
        _render_insights(result)
    else:
        st.info("Run an analysis from the sidebar to see insights.")
