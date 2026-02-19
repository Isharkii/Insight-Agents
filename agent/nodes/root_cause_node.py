"""
agent/nodes/root_cause_node.py

Root Cause Node: selects a business-specific root cause engine from
state.business_type and stores analysis in state.root_cause.
"""

from __future__ import annotations

from typing import Any

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
        supported = ", ".join(sorted(_ENGINE_AND_KPI_BY_BUSINESS_TYPE))
        raise ValueError(
            f"Unsupported business_type '{business_type}'. "
            f"Supported values: {supported}."
        )
    return selected


def root_cause_node(state: AgentState) -> AgentState:
    """
    LangGraph node: identify root causes with the business-specific engine.
    """
    business_type = str(state.get("business_type") or "")
    engine_used = business_type.lower() or "unknown"

    try:
        engine, kpi_key, engine_used = _select_engine_and_kpi_key(business_type)

        kpi_payload = state.get(kpi_key)
        if not isinstance(kpi_payload, dict):
            raise ValueError(f"Missing or invalid state['{kpi_key}'] payload.")

        forecast_payload = state.get("forecast_data")
        if not isinstance(forecast_payload, dict):
            raise ValueError("Missing or invalid state['forecast_data'] payload.")

        risk_data = state.get("risk_data")
        if not isinstance(risk_data, dict):
            raise ValueError("Missing or invalid state['risk_data'] payload.")

        flat_signals = normalize_signals(
            kpi_payload=kpi_payload,
            forecast_payload=forecast_payload,
        )

        result: dict[str, Any] = engine.analyze(
            kpi_data=flat_signals,
            forecast_data=flat_signals,
            risk_data=risk_data,
        )
        root_cause: dict[str, Any] = {**result, "engine_used": engine_used}

    except Exception as exc:  # noqa: BLE001
        root_cause = {
            "root_causes": [],
            "evidence": [],
            "impact": None,
            "confidence": 0.0,
            "recommended_action": None,
            "engine_used": engine_used,
            "error": str(exc),
        }

    return {**state, "root_cause": root_cause}
