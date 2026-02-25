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

    kpi_data: Optional[dict]
    saas_kpi_data: Optional[dict]
    ecommerce_kpi_data: Optional[dict]
    agency_kpi_data: Optional[dict]

    forecast_data: Optional[dict]
    risk_data: Optional[dict]
    root_cause: Optional[dict]
    segmentation: Optional[dict]
    prioritization: Optional[dict]
    pipeline_status: Optional[str]
    envelope_diagnostics: Optional[dict]
    dataset_confidence: Optional[float]
    ingestion_provenance: Optional[dict]
    ingestion_warnings: Optional[list[str]]
    growth_data: Optional[dict]
    timeseries_factors_data: Optional[dict]
    cohort_data: Optional[dict]
    category_formula_data: Optional[dict]
    multivariate_scenario_data: Optional[dict]
    final_response: Optional[str]
