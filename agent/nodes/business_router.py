"""
agent/nodes/business_router.py

Business Router Node: routes the graph to the correct KPI fetch branch
based on state.business_type.

No KPI logic, no DB access, deterministic only.
"""

from agent.state import AgentState

_ROUTE_MAP: dict[str, str] = {
    "saas":      "saas_kpi_fetch",
    "ecommerce": "ecommerce_kpi_fetch",
    "agency":    "agency_kpi_fetch",
    "general_timeseries": "kpi_fetch",
    "generic_timeseries": "kpi_fetch",
}


def business_router_node(state: AgentState) -> AgentState:
    """
    LangGraph node: pass-through that preserves state.

    Routing logic lives in ``route_by_business_type`` which is used
    as the condition function for ``add_conditional_edges``.

    Args:
        state: Current agent state.

    Returns:
        The unchanged state dictionary.
    """
    return state


def route_by_business_type(state: AgentState) -> str:
    """
    LangGraph conditional-edge function: resolve the next node name
    from state.business_type.

    Args:
        state: Current agent state.

    Returns:
        One of "saas_kpi_fetch", "ecommerce_kpi_fetch", "agency_kpi_fetch", "kpi_fetch".

    Unsupported values fall back to the generic KPI branch to keep the graph
    non-throwing.
    """
    business_type: str = str(state.get("business_type", "")).lower()
    route: str | None = _ROUTE_MAP.get(business_type)
    if route is None:
        # Non-throwing fallback keeps graph execution resilient.
        return "kpi_fetch"
    return route
