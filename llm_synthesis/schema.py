"""Canonical structured output schema for reasoning results."""

from pydantic import BaseModel


class InsightOutput(BaseModel):
    """Only allowed output contract for the reasoning layer."""

    insight: str
    evidence: str
    impact: str
    recommended_action: str
    priority: str
    confidence_score: float
