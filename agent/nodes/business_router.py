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
}


def business_router_node(state: AgentState) -> str:
    """
    LangGraph conditional edge function: resolve the next node name
    from state.business_type.

    Args:
        state: Current agent state.

    Returns:
        One of "saas_kpi_fetch", "ecommerce_kpi_fetch", "agency_kpi_fetch".

    Raises:
        ValueError: If state.business_type is not a supported value.
    """
    business_type: str = state.get("business_type", "")
    route: str | None = _ROUTE_MAP.get(business_type)

    if route is None:
        supported = ", ".join(f'"{k}"' for k in _ROUTE_MAP)
        raise ValueError(
            f"Unsupported business_type '{business_type}'. "
            f"Supported values: {supported}."
        )

    return route
