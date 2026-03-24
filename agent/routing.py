"""
agent/routing.py

All conditional-edge routing functions for the LangGraph workflow.
Extracted from graph.py and business_router.py to centralise routing decisions.
"""

from __future__ import annotations

from agent.state import AgentState

# ---------------------------------------------------------------------------
# Business-type routing (after intent → business_router)
# ---------------------------------------------------------------------------

_BUSINESS_TYPE_ROUTE_MAP: dict[str, str] = {
    "saas": "saas_kpi_fetch",
    "ecommerce": "ecommerce_kpi_fetch",
    "agency": "agency_kpi_fetch",
    "general_timeseries": "kpi_fetch",
    "generic_timeseries": "kpi_fetch",
}


def route_by_business_type(state: AgentState) -> str:
    """Resolve the KPI-fetch node from ``state.business_type``.

    Unsupported values fall back to the generic KPI branch.
    """
    business_type = str(state.get("business_type", "")).lower()
    return _BUSINESS_TYPE_ROUTE_MAP.get(business_type, "kpi_fetch")


# ---------------------------------------------------------------------------
# Synthesis gate routing (after synthesis_gate → llm | END)
# ---------------------------------------------------------------------------


def route_after_synthesis_gate(state: AgentState) -> str:
    """Route to LLM or short-circuit to END when synthesis is blocked."""
    if state.get("synthesis_blocked"):
        return "end"
    return "llm"
