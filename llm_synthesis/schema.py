"""Canonical structured output schema for reasoning results."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ConfidenceAdjustment(BaseModel):
    """Machine-readable confidence penalty entry."""

    model_config = ConfigDict(frozen=True)

    signal: str = Field(min_length=1)
    delta: float = Field(le=0.0, ge=-1.0)
    reason: str = Field(min_length=1)


class EnvelopeDiagnostics(BaseModel):
    """Partial-state diagnostics surfaced from upstream envelopes."""

    model_config = ConfigDict(frozen=True)

    warnings: list[str] = Field(default_factory=list)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    missing_signal: list[str] = Field(default_factory=list)
    confidence_adjustments: list[ConfidenceAdjustment] = Field(default_factory=list)


class InsightOutput(BaseModel):
    """Only allowed output contract for the reasoning layer."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    insight: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    impact: str = Field(min_length=1)
    recommended_action: str = Field(min_length=1)
    priority: Literal["low", "medium", "high", "critical"]
    confidence_score: float = Field(strict=True, ge=0.0, le=1.0)
    pipeline_status: Literal["success", "partial", "failed"] = "partial"
    diagnostics: EnvelopeDiagnostics | None = None

    @classmethod
    def failure(
        cls,
        reason: str,
        pipeline_status: Literal["success", "partial", "failed"] = "failed",
    ) -> "InsightOutput":
        return cls(
            insight="Analysis incomplete",
            evidence=reason,
            impact="Unable to compute reliable impact due to upstream validation failure.",
            recommended_action="Investigate upstream data or signal pipeline.",
            priority="high",
            confidence_score=0.0,
            pipeline_status=pipeline_status,
        )


FinalInsightResponse = InsightOutput
