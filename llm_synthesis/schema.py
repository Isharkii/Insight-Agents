"""Canonical structured output schema for reasoning results."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


_COMPETITOR_TERMS = ("competitor", "peer", "benchmark", "market")
_METRIC_TERMS = (
    "mrr",
    "arr",
    "growth",
    "churn",
    "arpu",
    "ltv",
    "retention",
    "risk",
    "share",
    "volatility",
    "rate",
    "metric",
    "kpi",
)
_EXPLICIT_RECOMMENDATION_TERMS = ("competitor", "gap", "strength", "weakness")
_LOW_CONFIDENCE_TONE_TERMS = (
    "conditional",
    "suggest",
    "appears",
    "may",
    "might",
    "could",
    "uncertain",
    "likely",
)
_GENERIC_ADVICE_PATTERNS = (
    "improve performance",
    "focus on growth",
    "monitor closely",
    "optimize operations",
    "align stakeholders",
    "take action",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _has_any_term(text: str, terms: tuple[str, ...]) -> bool:
    normalized = _normalize(text)
    return any(term in normalized for term in terms)


def _is_conditional(text: str) -> bool:
    return _normalize(text).startswith("conditional:")


class ConfidenceAdjustment(BaseModel):
    """Machine-readable confidence penalty entry (diagnostic utility type)."""

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


class CompetitiveAnalysis(BaseModel):
    """Competitor-only structured analysis block."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    summary: str = Field(min_length=1)
    market_position: str = Field(min_length=1)
    relative_performance: str = Field(min_length=1)
    key_advantages: list[str] = Field(default_factory=list, min_length=1)
    key_vulnerabilities: list[str] = Field(default_factory=list, min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_competitor_and_metric_references(self) -> "CompetitiveAnalysis":
        body = " ".join(
            [
                self.summary,
                self.market_position,
                self.relative_performance,
                *self.key_advantages,
                *self.key_vulnerabilities,
            ]
        )
        if not _has_any_term(body, _COMPETITOR_TERMS):
            raise ValueError(
                "competitive_analysis must reference competitor context "
                "(competitor/peer/benchmark/market)."
            )
        if not _has_any_term(body, _METRIC_TERMS):
            raise ValueError(
                "competitive_analysis must reference measurable metrics "
                "(e.g., growth/churn/mrr/risk/share)."
            )
        return self


class StrategicRecommendations(BaseModel):
    """Structured recommendation block linked to competitor analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    immediate_actions: list[str] = Field(default_factory=list, min_length=1)
    mid_term_moves: list[str] = Field(default_factory=list, min_length=1)
    defensive_strategies: list[str] = Field(default_factory=list, min_length=1)
    offensive_strategies: list[str] = Field(default_factory=list, min_length=1)

    @model_validator(mode="after")
    def _validate_non_generic_explicit_and_unique(self) -> "StrategicRecommendations":
        all_items = [
            *self.immediate_actions,
            *self.mid_term_moves,
            *self.defensive_strategies,
            *self.offensive_strategies,
        ]
        dedupe: set[str] = set()
        for item in all_items:
            normalized = _normalize(item)
            if normalized in dedupe:
                raise ValueError("Recommendations must not repeat across sections.")
            dedupe.add(normalized)

            if not _has_any_term(normalized, _EXPLICIT_RECOMMENDATION_TERMS):
                raise ValueError(
                    "Each recommendation must explicitly reference competitor gaps, "
                    "strengths, or weaknesses."
                )
            if any(pattern in normalized for pattern in _GENERIC_ADVICE_PATTERNS):
                raise ValueError("Generic recommendations are not allowed.")
        return self


class InsightOutput(BaseModel):
    """Only allowed output contract for the reasoning layer."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    competitive_analysis: CompetitiveAnalysis
    strategic_recommendations: StrategicRecommendations

    @property
    def confidence_score(self) -> float:
        return float(self.competitive_analysis.confidence)

    @property
    def insight(self) -> str:
        return self.competitive_analysis.summary

    @property
    def evidence(self) -> str:
        return self.competitive_analysis.relative_performance

    @property
    def impact(self) -> str:
        return self.competitive_analysis.market_position

    @property
    def recommended_action(self) -> str:
        first = self.strategic_recommendations.immediate_actions
        return first[0] if first else ""

    @property
    def priority(self) -> Literal["low", "medium", "high", "critical"]:
        confidence = self.competitive_analysis.confidence
        if confidence >= 0.8:
            return "high"
        if confidence >= 0.6:
            return "medium"
        if confidence >= 0.4:
            return "low"
        return "critical"

    @property
    def pipeline_status(self) -> Literal["success", "partial", "failed"]:
        confidence = self.competitive_analysis.confidence
        if confidence <= 0.0:
            return "failed"
        if confidence < 0.8:
            return "partial"
        return "success"

    @property
    def diagnostics(self) -> None:
        return None

    @model_validator(mode="after")
    def _validate_confidence_tone_and_conditional(self) -> "InsightOutput":
        confidence = float(self.competitive_analysis.confidence)
        recommendations = [
            *self.strategic_recommendations.immediate_actions,
            *self.strategic_recommendations.mid_term_moves,
            *self.strategic_recommendations.defensive_strategies,
            *self.strategic_recommendations.offensive_strategies,
        ]
        analysis_text = " ".join(
            [
                self.competitive_analysis.summary,
                self.competitive_analysis.market_position,
                self.competitive_analysis.relative_performance,
            ]
        )

        if confidence < 0.5:
            if not any(_is_conditional(item) for item in recommendations):
                raise ValueError(
                    'When confidence < 0.5, recommendations must be labeled as "conditional".'
                )
            if not _has_any_term(analysis_text, _LOW_CONFIDENCE_TONE_TERMS):
                raise ValueError(
                    "Low-confidence analysis must use cautious/conditional tone."
                )
        elif confidence >= 0.8 and _has_any_term(
            analysis_text,
            ("highly uncertain", "insufficient evidence", "unable to determine"),
        ):
            raise ValueError(
                "High-confidence analysis should avoid strongly uncertain tone."
            )
        return self

    @classmethod
    def failure(
        cls,
        reason: str,
        pipeline_status: Literal["success", "partial", "failed"] = "failed",
    ) -> "InsightOutput":
        del pipeline_status  # Preserved for backward call compatibility.
        return cls(
            competitive_analysis=CompetitiveAnalysis(
                summary=(
                    "Conditional: competitor benchmark analysis could not be completed "
                    "due to missing competitor metrics."
                ),
                market_position=(
                    "Conditional: market position remains uncertain relative to competitors."
                ),
                relative_performance=(
                    f"Conditional: competitor gap/strength/weakness assessment blocked ({reason})."
                ),
                key_advantages=[
                    "Conditional: no validated competitor strength advantage can be confirmed."
                ],
                key_vulnerabilities=[
                    "Conditional: competitor data gaps are a weakness in current assessment reliability."
                ],
                confidence=0.0,
            ),
            strategic_recommendations=StrategicRecommendations(
                immediate_actions=[
                    "Conditional: close competitor metric gaps before acting on strength/weakness assumptions."
                ],
                mid_term_moves=[
                    "Conditional: build benchmark coverage to validate competitor weaknesses against metric trends."
                ],
                defensive_strategies=[
                    "Conditional: protect against competitor strength where risk and churn metrics are incomplete."
                ],
                offensive_strategies=[
                    "Conditional: target competitor weakness only after benchmark gaps are resolved."
                ],
            ),
        )


FinalInsightResponse = InsightOutput
