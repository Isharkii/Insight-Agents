"""
agent/nodes/root_cause_node.py

Root Cause Node: selects a business-specific root cause engine from
state.business_type and stores analysis in state.root_cause.
"""

from __future__ import annotations

from typing import Any

from agent.nodes.node_result import failed, payload_of, skipped, status_of, success
from agent.signal_normalizer import normalize_signals
from agent.state import AgentState
from root_cause.agency_rules import AgencyRootCauseEngine
from root_cause.ecommerce_rules import EcommerceRootCauseEngine
from root_cause.saas_rules import SaaSRootCauseEngine

_SAAS_ENGINE = SaaSRootCauseEngine()
_ECOMMERCE_ENGINE = EcommerceRootCauseEngine()
_AGENCY_ENGINE = AgencyRootCauseEngine()

_ENGINE_AND_KPI_BY_BUSINESS_TYPE: dict[str, tuple[Any, str, str]] = {
    "saas": (_SAAS_ENGINE, "saas_kpi_data", "saas"),
    "ecommerce": (_ECOMMERCE_ENGINE, "ecommerce_kpi_data", "ecommerce"),
    "agency": (_AGENCY_ENGINE, "agency_kpi_data", "agency"),
}


def _select_engine_and_kpi_key(business_type: str) -> tuple[Any, str, str]:
    key = business_type.lower()
    selected = _ENGINE_AND_KPI_BY_BUSINESS_TYPE.get(key)
    if selected is None:
        return None, "", "unknown"
    return selected


def root_cause_node(state: AgentState) -> AgentState:
    """
    LangGraph node: identify root causes with the business-specific engine.
    """
    business_type = str(state.get("business_type") or "")
    engine_used = business_type.lower() or "unknown"

    engine, kpi_key, engine_used = _select_engine_and_kpi_key(business_type)
    if engine is None or not kpi_key:
        root_cause = skipped(
            "unsupported_business_type",
            {"business_type": business_type},
        )
        return {**state, "root_cause": root_cause}

    kpi_state = state.get(kpi_key)
    kpi_payload = payload_of(kpi_state)
    if status_of(kpi_state) != "success" or not isinstance(kpi_payload, dict):
        root_cause = skipped(
            "kpi_unavailable",
            {"engine_used": engine_used},
        )
        return {**state, "root_cause": root_cause}

    forecast_state = state.get("forecast_data")
    forecast_payload = payload_of(forecast_state)
    if status_of(forecast_state) != "success" or not isinstance(forecast_payload, dict):
        root_cause = skipped(
            "forecast_unavailable",
            {"engine_used": engine_used},
        )
        return {**state, "root_cause": root_cause}

    risk_state = state.get("risk_data")
    risk_data = payload_of(risk_state)
    if status_of(risk_state) != "success" or not isinstance(risk_data, dict):
        root_cause = skipped(
            "risk_unavailable",
            {"engine_used": engine_used},
        )
        return {**state, "root_cause": root_cause}

    try:
        flat_signals = normalize_signals(
            kpi_payload=kpi_payload,
            forecast_payload=forecast_payload,
        )

        result: dict[str, Any] = engine.analyze(
            kpi_data=flat_signals,
            forecast_data=flat_signals,
            risk_data=risk_data,
        )
        payload: dict[str, Any] = {**result, "engine_used": engine_used}
        root_cause = success(payload)

    except Exception as exc:  # noqa: BLE001
        root_cause = failed(
            str(exc),
            {"engine_used": engine_used},
        )

    return {**state, "root_cause": root_cause}
