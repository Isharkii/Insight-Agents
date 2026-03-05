"""
agent/state.py

LangGraph agent state schema for the Insight Agent.
Supports multiple business model KPI namespaces.
"""

from typing import Literal, Optional
from typing_extensions import TypedDict


class CompetitiveContextMetric(TypedDict, total=False):
    """Numeric-only competitive signal contract (safe for prompts)."""

    metric_name: str
    unit: str
    sample_size: int
    mean: float | None
    median: float | None
    min_value: float | None
    max_value: float | None
    stdev: float | None


class CompetitiveContext(TypedDict, total=False):
    """Structured competitive context emitted before the LLM node."""

    available: bool
    source: Literal[
        "deterministic_local",
        "external_fetch",
        "disabled",
        "unavailable",
    ]
    peer_count: int
    peers: list[str]
    metrics: list[str]
    benchmark_rows_count: int
    numeric_signals: list[CompetitiveContextMetric]
    cache_hit: bool
    generated_at: str
    warnings: list[str]


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
    competitive_context: Optional[CompetitiveContext]
    signal_integrity: Optional[dict]
    synthesis_blocked: Optional[bool]
    final_response: Optional[str]
