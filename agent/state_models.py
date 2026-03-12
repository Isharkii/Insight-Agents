"""
agent/state_models.py

Strongly typed Pydantic contracts for LangGraph state and node envelopes.
"""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

NodeStatus = Literal["success", "insufficient_data", "skipped", "failed"]
PipelineStatus = Literal["success", "partial", "failed"]
T = TypeVar("T")


class NodeEnvelope(BaseModel, Generic[T]):
    """Typed node envelope shared across graph state keys."""

    model_config = ConfigDict(extra="forbid")

    status: NodeStatus
    payload: T | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_status_semantics(self) -> "NodeEnvelope[T]":
        if self.status == "failed" and not self.errors:
            if not isinstance(self.payload, dict) or not (
                self.payload.get("error") or self.payload.get("reason")
            ):
                raise ValueError(
                    "failed envelopes must include `errors` or payload.error/payload.reason"
                )
        return self


class CompetitiveContextMetricModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_name: str
    unit: str
    sample_size: int
    mean: float | None = None
    median: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    stdev: float | None = None


class CompetitiveContextModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    source: Literal[
        "deterministic_local",
        "external_fetch",
        "disabled",
        "unavailable",
    ]
    peer_count: int
    peers: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    benchmark_rows_count: int = 0
    numeric_signals: list[CompetitiveContextMetricModel] = Field(default_factory=list)
    cache_hit: bool = False
    generated_at: str | None = None
    warnings: list[str] = Field(default_factory=list)


class KpiPayloadModel(BaseModel):
    """
    Typed KPI payload contract.

    Supports compact and legacy payload formats by allowing extra fields.
    """

    model_config = ConfigDict(extra="allow")

    state_mode: Literal["derived_only"] | None = None
    record_ref: str | None = None
    record_count: int = Field(default=0, ge=0)
    fetched_for: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    metrics: list[str] = Field(default_factory=list)
    metric_series: dict[str, list[float]] = Field(default_factory=dict)
    latest_computed_kpis: dict[str, Any] = Field(default_factory=dict)


class ForecastPayloadModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    forecasts: dict[str, Any] = Field(default_factory=dict)
    fetched_for: str | None = None
    metrics_queried: list[str] = Field(default_factory=list)
    confidence_breakdown: dict[str, Any] | None = None


class RiskPayloadModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    entity_name: str | None = None
    risk_score: float | int | None = None
    risk_level: str | None = None
    confidence_breakdown: dict[str, Any] | None = None
    conflict_metadata: dict[str, Any] | None = None


class AgentStateModel(BaseModel):
    """Canonical, strongly typed graph state contract."""

    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None
    user_query: str
    business_type: str
    entity_name: str

    kpi_data: NodeEnvelope[KpiPayloadModel] | None = None
    saas_kpi_data: NodeEnvelope[KpiPayloadModel] | None = None
    ecommerce_kpi_data: NodeEnvelope[KpiPayloadModel] | None = None
    agency_kpi_data: NodeEnvelope[KpiPayloadModel] | None = None

    forecast_data: NodeEnvelope[ForecastPayloadModel] | None = None
    risk_data: NodeEnvelope[RiskPayloadModel | dict[str, Any]] | None = None
    root_cause: NodeEnvelope[dict[str, Any]] | None = None
    segmentation: NodeEnvelope[dict[str, Any]] | None = None
    prioritization: dict[str, Any] | None = None
    pipeline_status: PipelineStatus | None = None
    envelope_diagnostics: dict[str, Any] | None = None
    dataset_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    ingestion_provenance: dict[str, Any] | None = None
    ingestion_warnings: list[str] | None = None
    growth_data: NodeEnvelope[dict[str, Any]] | None = None
    timeseries_factors_data: NodeEnvelope[dict[str, Any]] | None = None
    cohort_data: NodeEnvelope[dict[str, Any]] | None = None
    category_formula_data: NodeEnvelope[dict[str, Any]] | None = None
    unit_economics_data: NodeEnvelope[dict[str, Any]] | None = None
    multivariate_scenario_data: NodeEnvelope[dict[str, Any]] | None = None
    signal_conflicts: NodeEnvelope[dict[str, Any]] | None = None
    signal_enrichment: NodeEnvelope[dict[str, Any]] | None = None
    competitive_context: CompetitiveContextModel | None = None
    signal_integrity: dict[str, Any] | None = None
    synthesis_blocked: bool | None = None
    final_response: str | None = None
    llm_model_override: str | None = None

    @classmethod
    def from_graph_state(cls, state: dict[str, Any]) -> "AgentStateModel":
        return cls.model_validate(state)

    def to_graph_state(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude_none=True)


class AgentStatePatchModel(BaseModel):
    """
    Partial state model for node I/O validation.

    Every key is optional, but any provided key must be correctly typed.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None
    user_query: str | None = None
    business_type: str | None = None
    entity_name: str | None = None

    kpi_data: NodeEnvelope[KpiPayloadModel] | None = None
    saas_kpi_data: NodeEnvelope[KpiPayloadModel] | None = None
    ecommerce_kpi_data: NodeEnvelope[KpiPayloadModel] | None = None
    agency_kpi_data: NodeEnvelope[KpiPayloadModel] | None = None

    forecast_data: NodeEnvelope[ForecastPayloadModel] | None = None
    risk_data: NodeEnvelope[RiskPayloadModel | dict[str, Any]] | None = None
    root_cause: NodeEnvelope[dict[str, Any]] | None = None
    segmentation: NodeEnvelope[dict[str, Any]] | None = None
    prioritization: dict[str, Any] | None = None
    pipeline_status: PipelineStatus | None = None
    envelope_diagnostics: dict[str, Any] | None = None
    dataset_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    ingestion_provenance: dict[str, Any] | None = None
    ingestion_warnings: list[str] | None = None
    growth_data: NodeEnvelope[dict[str, Any]] | None = None
    timeseries_factors_data: NodeEnvelope[dict[str, Any]] | None = None
    cohort_data: NodeEnvelope[dict[str, Any]] | None = None
    category_formula_data: NodeEnvelope[dict[str, Any]] | None = None
    unit_economics_data: NodeEnvelope[dict[str, Any]] | None = None
    multivariate_scenario_data: NodeEnvelope[dict[str, Any]] | None = None
    signal_conflicts: NodeEnvelope[dict[str, Any]] | None = None
    signal_enrichment: NodeEnvelope[dict[str, Any]] | None = None
    competitive_context: CompetitiveContextModel | None = None
    signal_integrity: dict[str, Any] | None = None
    synthesis_blocked: bool | None = None
    final_response: str | None = None
    llm_model_override: str | None = None

