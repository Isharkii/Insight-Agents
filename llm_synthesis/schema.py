"""Canonical structured output schema for reasoning results."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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

    @classmethod
    def failure(cls, reason: str) -> "InsightOutput":
        return cls(
            insight="Analysis incomplete",
            evidence=reason,
            impact="Unable to compute reliable impact due to upstream validation failure.",
            recommended_action="Investigate upstream data or signal pipeline.",
            priority="high",
            confidence_score=0.0,
        )


FinalInsightResponse = InsightOutput
