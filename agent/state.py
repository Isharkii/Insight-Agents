"""
agent/state.py

LangGraph agent state schema for the Insight Agent.
Supports multiple business model KPI namespaces.
"""

from typing import Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Shared state passed between all nodes in the LangGraph agent graph."""

    user_query: str
    business_type: str
    entity_name: str

    saas_kpi_data: Optional[dict]
    ecommerce_kpi_data: Optional[dict]
    agency_kpi_data: Optional[dict]

    forecast_data: Optional[dict]
    risk_data: Optional[dict]
    root_cause: Optional[dict]
    segmentation: Optional[dict]
    prioritization: Optional[dict]
    final_response: Optional[str]
