"""Pydantic schema for LLM synthesis structured output."""

from typing import List

from pydantic import BaseModel, Field


class SynthesisOutput(BaseModel):
    """Strict schema for LLM synthesis output.

    All fields are required. Used to validate and enforce
    structured responses from the reasoning layer.
    """

    executive_summary: str = Field(
        ...,
        description="High-level summary of the analysis",
    )
    key_findings: List[str] = Field(
        ...,
        description="List of key findings from the analysis",
    )
    primary_risk: str = Field(
        ...,
        description="The most critical risk identified",
    )
    recommended_actions: List[str] = Field(
        ...,
        description="Actionable recommendations based on findings",
    )
    priority_level: str = Field(
        ...,
        description="Priority level (e.g. Critical, High, Medium, Low)",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1",
    )
