"""
agent/graph.py

LangGraph workflow assembly for the Insight Agent.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.failure_codes import CRITICAL_FAILURES, OPTIONAL_FAILURES
from agent.state import AgentState
from agent.nodes.business_router import business_router_node
from agent.nodes.intent import intent_node
from agent.nodes.saas_kpi_node import saas_kpi_fetch_node
from agent.nodes.ecommerce_kpi_node import ecommerce_kpi_fetch_node
from agent.nodes.agency_kpi_node import agency_kpi_fetch_node
from agent.nodes.forecast_node import forecast_fetch_node
from agent.nodes.risk_node import risk_node
from agent.nodes.root_cause_node import root_cause_node
from agent.nodes.segmentation_node import segmentation_node
from agent.nodes.prioritization_node import prioritization_node
from agent.nodes.llm_node import llm_node

logger = logging.getLogger(__name__)


def _handle_pipeline_failure(
    *,
    failure_code: str,
    message: str,
    stage_name: str,
    missing_fields: list[str],
    business_type: str,
) -> None:
    payload = {
        "stage_name": stage_name,
        "missing_fields": missing_fields,
        "business_type": business_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if failure_code in CRITICAL_FAILURES:
        logger.error(
            "Pipeline critical failure code=%s message=%s",
            failure_code,
            message,
            extra=payload,
        )
        raise ValueError(message)
    if failure_code in OPTIONAL_FAILURES:
        logger.warning(
            "Pipeline optional failure code=%s message=%s",
            failure_code,
            message,
            extra=payload,
        )
        return
    raise ValueError(f"Unknown pipeline failure code '{failure_code}': {message}")


def validate_core_signals(
    kpi_payload: dict[str, Any] | None,
    forecast_payload: dict[str, Any] | None,
    risk_payload: dict[str, Any] | None,
    *,
    stage_name: str,
    business_type: str,
) -> None:
    """Validate required core signals before critical downstream nodes."""
    records = (
        kpi_payload.get("records")
        if isinstance(kpi_payload, dict)
        else None
    )
    if not isinstance(records, list) or not records:
        _handle_pipeline_failure(
            failure_code="empty_kpi",
            message="Missing KPI records in core signal context.",
            stage_name=stage_name,
            missing_fields=["kpi.records"],
            business_type=business_type,
        )

    forecasts = (
        forecast_payload.get("forecasts")
        if isinstance(forecast_payload, dict)
        else None
    )
    if not isinstance(forecasts, dict) or not forecasts:
        _handle_pipeline_failure(
            failure_code="empty_forecast",
            message="Missing forecast entries in core signal context.",
            stage_name=stage_name,
            missing_fields=["forecast.forecasts"],
            business_type=business_type,
        )

    has_forecast_entry = any(
        isinstance(entry, dict) and isinstance(entry.get("forecast_data"), dict)
        for entry in forecasts.values()
    )
    if not has_forecast_entry:
        _handle_pipeline_failure(
            failure_code="empty_forecast",
            message="Forecast entries are present but contain no usable forecast_data.",
            stage_name=stage_name,
            missing_fields=["forecast.forecast_data"],
            business_type=business_type,
        )

    risk_score = risk_payload.get("risk_score") if isinstance(risk_payload, dict) else None
    if risk_score is None:
        _handle_pipeline_failure(
            failure_code="empty_risk",
            message="Missing computed risk_score in core signal context.",
            stage_name=stage_name,
            missing_fields=["risk.risk_score"],
            business_type=business_type,
        )


def _resolve_kpi_payload(state: AgentState) -> dict[str, Any]:
    kpi_key_by_business_type = {
        "saas": "saas_kpi_data",
        "ecommerce": "ecommerce_kpi_data",
        "agency": "agency_kpi_data",
    }
    business_type = str(state.get("business_type") or "").lower()
    primary_key = kpi_key_by_business_type.get(business_type)
    if primary_key:
        payload = state.get(primary_key)
        if isinstance(payload, dict):
            return payload

    for key in ("saas_kpi_data", "ecommerce_kpi_data", "agency_kpi_data"):
        payload = state.get(key)
        if isinstance(payload, dict):
            return payload
    return {}


def _validate_core_signals_before_root_cause(state: AgentState) -> AgentState:
    business_type = str(state.get("business_type") or "")
    validate_core_signals(
        _resolve_kpi_payload(state),
        state.get("forecast_data"),
        state.get("risk_data"),
        stage_name="root_cause_precheck",
        business_type=business_type,
    )
    return state


def _validate_core_signals_before_llm(state: AgentState) -> AgentState:
    business_type = str(state.get("business_type") or "")
    validate_core_signals(
        _resolve_kpi_payload(state),
        state.get("forecast_data"),
        state.get("risk_data"),
        stage_name="llm_precheck",
        business_type=business_type,
    )
    return state


def build_graph():
    """
    Build and compile the Insight Agent LangGraph workflow.
    """
    graph = StateGraph(AgentState)

    graph.add_node("intent", intent_node)
    graph.add_node("business_router", business_router_node)
    graph.add_node("saas_kpi_fetch", saas_kpi_fetch_node)
    graph.add_node("ecommerce_kpi_fetch", ecommerce_kpi_fetch_node)
    graph.add_node("agency_kpi_fetch", agency_kpi_fetch_node)
    graph.add_node("forecast_fetch", forecast_fetch_node)
    graph.add_node("risk", risk_node)
    graph.add_node(
        "validate_core_signals_before_root_cause",
        _validate_core_signals_before_root_cause,
    )
    graph.add_node("root_cause", root_cause_node)
    graph.add_node("segmentation", segmentation_node)
    graph.add_node("prioritization", prioritization_node)
    graph.add_node(
        "validate_core_signals_before_llm",
        _validate_core_signals_before_llm,
    )
    graph.add_node("llm", llm_node)

    graph.add_edge(START, "intent")
    graph.add_edge("intent", "business_router")
    graph.add_conditional_edges(
        "business_router",
        business_router_node,
        {
            "saas_kpi_fetch": "saas_kpi_fetch",
            "ecommerce_kpi_fetch": "ecommerce_kpi_fetch",
            "agency_kpi_fetch": "agency_kpi_fetch",
        },
    )
    graph.add_edge("saas_kpi_fetch", "forecast_fetch")
    graph.add_edge("ecommerce_kpi_fetch", "forecast_fetch")
    graph.add_edge("agency_kpi_fetch", "forecast_fetch")
    graph.add_edge("forecast_fetch", "risk")
    graph.add_edge("risk", "validate_core_signals_before_root_cause")
    graph.add_edge("validate_core_signals_before_root_cause", "root_cause")
    graph.add_edge("root_cause", "segmentation")
    graph.add_edge("segmentation", "prioritization")
    graph.add_edge("prioritization", "validate_core_signals_before_llm")
    graph.add_edge("validate_core_signals_before_llm", "llm")
    graph.add_edge("llm", END)

    return graph.compile()


insight_graph = build_graph()
