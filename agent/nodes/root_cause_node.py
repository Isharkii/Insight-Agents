"""
agent/nodes/root_cause_node.py

Root Cause Node: delegates analysis to RootCauseOrchestrator using
state.kpi_data, state.forecast_data, and state.risk_data.

No rule logic, no DB access.
"""

from __future__ import annotations

from typing import Any

from agent.state import AgentState
from root_cause.orchestrator import RootCauseOrchestrator

# ---------------------------------------------------------------------------
# Business-type normaliser
# ---------------------------------------------------------------------------
# RootCauseOrchestrator only knows "saas", "ecommerce", and "agency".
# Map intent-node types that are semantically equivalent.

_TYPE_MAP: dict[str, str] = {
    "saas":         "saas",
    "software":     "saas",
    "retail":       "ecommerce",
    "ecommerce":    "ecommerce",
    "food_service": "ecommerce",
    "agency":       "agency",
    "marketing":    "agency",
    "consulting":   "agency",
}

_FALLBACK_TYPE = "ecommerce"

_orchestrator = RootCauseOrchestrator()


def _resolve_type(business_type: str) -> str:
    """Map an intent-node business_type to a supported engine key."""
    return _TYPE_MAP.get(business_type.lower(), _FALLBACK_TYPE)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def root_cause_node(state: AgentState) -> AgentState:
    """
    LangGraph node: identify root causes for the entity's performance signals.

    Reads:
        state["business_type"]  — resolved to a supported engine type.
        state["kpi_data"]       — output of kpi_fetch_node.
        state["forecast_data"]  — output of forecast_fetch_node.
        state["risk_data"]      — output of risk_node.

    Writes:
        state["root_cause"] — dict produced by the engine:
            "root_causes"        : list[str]
            "evidence"           : list[str] | dict
            "impact"             : str | dict
            "confidence"         : float  (0–1)
            "recommended_action" : str
            "engine_used"        : str    (resolved business type)
            "error"              : str    (present only on failure)
    """
    business_type: str = state.get("business_type") or "general"
    kpi_data: dict = state.get("kpi_data") or {}
    forecast_data: dict = state.get("forecast_data") or {}
    risk_data: dict = state.get("risk_data") or {}

    resolved_type = _resolve_type(business_type)

    try:
        result: dict[str, Any] = _orchestrator.analyze(
            business_type=resolved_type,
            kpi_data=kpi_data,
            forecast_data=forecast_data,
            risk_data=risk_data,
        )
        root_cause: dict[str, Any] = {**result, "engine_used": resolved_type}

    except Exception as exc:  # noqa: BLE001
        root_cause = {
            "root_causes": [],
            "evidence": [],
            "impact": None,
            "confidence": 0.0,
            "recommended_action": None,
            "engine_used": resolved_type,
            "error": str(exc),
        }

    return {**state, "root_cause": root_cause}
