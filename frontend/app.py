"""Streamlit frontend for InsightAgent.

Replaceable UI layer — all display logic lives here.
Core agent logic is invoked via insight_graph only.
"""

from __future__ import annotations

import io
import json
from typing import Optional

import pandas as pd
from pydantic import ValidationError
import streamlit as st
from llm_synthesis.schema import InsightOutput

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="InsightAgent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Lazy graph import (avoids DB init cost until needed) ───────────────────
@st.cache_resource(show_spinner=False)
def _load_graph():
    from agent.graph import insight_graph  # noqa: PLC0415

    return insight_graph


# ── Session state defaults ─────────────────────────────────────────────────
_STATE_DEFAULTS: dict = {
    "result": None,
    "uploaded_df": None,
    "running": False,
}

for _key, _val in _STATE_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("InsightAgent")
    st.caption("Modular AI Insight Engine")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload data (CSV)",
        type=["csv"],
        help="Optional — preview your dataset before running analysis.",
    )
    if uploaded_file:
        df = pd.read_csv(io.BytesIO(uploaded_file.read()))
        st.session_state.uploaded_df = df
        st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    st.divider()
    user_query = st.text_area(
        "Query",
        placeholder="e.g. Analyse SaaS KPIs for Acme Cloud",
        height=100,
    )
    run = st.button("Run Analysis", type="primary", use_container_width=True)

    if st.session_state.result:
        if st.button("Clear", use_container_width=True):
            st.session_state.result = None
            st.session_state.uploaded_df = None
            st.rerun()


# ── Run graph ──────────────────────────────────────────────────────────────
if run and user_query.strip():
    with st.spinner("Running insight pipeline…"):
        try:
            graph = _load_graph()
            state = graph.invoke({"user_query": user_query.strip()})
            st.session_state.result = state
        except Exception as exc:  # noqa: BLE001
            st.error(f"Pipeline error: {exc}")
elif run:
    st.sidebar.warning("Please enter a query before running.")


# ── Helper renderers ───────────────────────────────────────────────────────
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


def _render_kpi_dashboard(result: dict) -> None:
    kpi: Optional[dict] = (
        result.get("saas_kpi_data")
        or result.get("ecommerce_kpi_data")
        or result.get("agency_kpi_data")
    )

    if not kpi:
        st.info("No KPI data available for this run.")
        return

    records = kpi.get("records", [])
    if not records:
        st.warning("KPI records are empty.")
        st.json(kpi)
        return

    # Flatten latest record's computed_kpis into metric cards
    latest = records[-1].get("computed_kpis", {})
    if not latest:
        st.json(kpi)
        return

    st.subheader("KPI Metrics")
    cols = st.columns(min(len(latest), 4))
    for idx, (name, val) in enumerate(latest.items()):
        display = f"{val:.2f}" if isinstance(val, float) else str(val)
        cols[idx % 4].metric(name.replace("_", " ").title(), display)

    with st.expander("Raw KPI payload"):
        st.json(kpi)


def _render_risk(result: dict) -> None:
    risk: Optional[dict] = result.get("risk_data")
    root: Optional[dict] = result.get("root_cause")

    if not risk:
        st.info("No risk data available.")
        return

    score: int = risk.get("risk_score", 0)
    level: str = risk.get("risk_level", "unknown").upper()

    _LEVEL_COLOUR = {
        "LOW": "🟢",
        "MODERATE": "🟡",
        "HIGH": "🟠",
        "CRITICAL": "🔴",
    }

    cols = st.columns([1, 3])
    with cols[0]:
        st.metric("Risk Score", score, help="0 = no risk, 100 = critical")
        st.markdown(f"**Level:** {_LEVEL_COLOUR.get(level, '⚪')} {level}")
    with cols[1]:
        st.progress(score / 100, text=f"{score} / 100")

    if root:
        st.subheader("Root Cause Analysis")
        causes = root.get("root_causes") or root.get("primary_issue") or "—"
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

    with st.expander("Raw risk & root-cause payload"):
        st.json({"risk_data": risk, "root_cause": root})


def _render_segments(result: dict) -> None:
    seg: Optional[dict] = result.get("segmentation")

    if not seg or not seg.get("found"):
        st.info("No segmentation data found for this entity.")
        return

    segment_data: dict = seg.get("segment_data", {})
    if not segment_data:
        st.json(seg)
        return

    st.subheader(f"Segments  —  {seg.get('n_clusters', '?')} clusters")
    for label, profile in segment_data.items():
        with st.expander(label.replace("_", " ").title()):
            if isinstance(profile, dict):
                rows = [{"Metric": k, "Value": v} for k, v in profile.items()]
                st.table(pd.DataFrame(rows))
            else:
                st.write(profile)


def _render_insights(result: dict) -> None:
    response: Optional[str] = result.get("final_response")

    if not response:
        st.info("No insight response available yet.")
        return

    # Try to parse as InsightOutput JSON; fall back to raw text
    parsed: Optional[InsightOutput] = None
    try:
        payload = json.loads(response)
        parsed = InsightOutput.model_validate(payload)
    except (json.JSONDecodeError, TypeError, ValidationError):
        pass

    if parsed:
        st.subheader("Insight")
        st.write(parsed.insight or "—")

        st.subheader("Evidence")
        st.write(parsed.evidence or "—")

        st.subheader("Impact")
        st.warning(parsed.impact or "—")

        st.subheader("Recommended Action")
        st.markdown(f"- {parsed.recommended_action or '—'}")

        st.subheader("Priority")
        st.write(parsed.priority or "—")

        conf = parsed.confidence_score
        if conf is not None:
            st.progress(float(conf), text=f"Confidence: {conf:.0%}")
    else:
        # Plain-text fallback (legacy llm_node responses)
        st.markdown(response)

    with st.expander("Raw agent state"):
        safe = {k: v for k, v in result.items() if k != "final_response"}
        st.json(safe)


# ── Main content area ──────────────────────────────────────────────────────
result: Optional[dict] = st.session_state.result

tab_upload, tab_kpi, tab_risk, tab_segments, tab_insights = st.tabs(
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
        st.info("Run an analysis from the sidebar to see segmentation data.")

with tab_insights:
    if result:
        _render_insights(result)
    else:
        st.info("Run an analysis from the sidebar to see insights.")
