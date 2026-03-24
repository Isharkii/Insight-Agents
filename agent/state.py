"""
agent/state.py

LangGraph agent state schema for the Insight Agent.
Supports multiple business model KPI namespaces.
"""

from typing import Annotated, Any, Literal, Optional
from typing_extensions import TypedDict


def _last_writer_wins(current: object, new: object) -> object:
    """Reducer: accept the latest value for keys written by multiple nodes."""
    return new


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
    news_highlights: list[dict[str, Any]]
    cache_hit: bool
    generated_at: str
    warnings: list[str]


class AgentState(TypedDict):
    """Shared state passed between all nodes in the LangGraph agent graph."""

    request_id: Annotated[Optional[str], _last_writer_wins]
    user_query: str
    business_type: str
    entity_name: str
    peer_entities: Optional[list[str]]

    kpi_data: Optional[dict]
    saas_kpi_data: Optional[dict]
    ecommerce_kpi_data: Optional[dict]
    agency_kpi_data: Optional[dict]

    forecast_data: Optional[dict]
    risk_data: Optional[dict]
    root_cause: Optional[dict]
    segmentation: Optional[dict]
    prioritization: Optional[dict]
    pipeline_status: Annotated[Optional[str], _last_writer_wins]
    envelope_diagnostics: Annotated[Optional[dict], _last_writer_wins]
    dataset_confidence: Annotated[Optional[float], _last_writer_wins]
    ingestion_provenance: Optional[dict]
    ingestion_warnings: Optional[list[str]]
    growth_data: Optional[dict]
    timeseries_factors_data: Optional[dict]
    cohort_data: Optional[dict]
    category_formula_data: Optional[dict]
    unit_economics_data: Optional[dict]
    multivariate_scenario_data: Optional[dict]
    benchmark_data: Optional[dict]
    signal_conflicts: Optional[dict]
    signal_enrichment: Optional[dict]
    competitive_context: Annotated[Optional[CompetitiveContext], _last_writer_wins]
    signal_integrity: Annotated[Optional[dict], _last_writer_wins]
    synthesis_blocked: Annotated[Optional[bool], _last_writer_wins]
    eligible_for_llm: Annotated[Optional[bool], _last_writer_wins]
    block_reasons: Annotated[Optional[list], _last_writer_wins]
    final_response: Annotated[Optional[str], _last_writer_wins]
    competitors: Optional[list[str]]
    self_analysis_only: Optional[bool]
    llm_model_override: Optional[str]
