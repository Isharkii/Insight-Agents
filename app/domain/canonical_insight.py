"""
app/domain/canonical_insight.py

Pydantic domain models used by CSV ingestion flow.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CanonicalInsightInput(BaseModel):
    """
    Typed canonical record prepared for persistence.

    Category is a free-form string — no hardcoded allowlist.
    The DB column is String(32); any value that fits is accepted.
    """

    model_config = ConfigDict(frozen=True)

    source_type: str
    entity_name: str
    category: str
    role: str | None = None
    metric_name: str
    metric_value: Any
    timestamp: datetime
    region: str | None = None
    metadata_json: dict[str, Any] | None = None


class RowValidationError(BaseModel):
    """
    One CSV row validation error detail.
    """

    model_config = ConfigDict(frozen=True)

    row_number: int
    message: str
    column: str | None = None
    value: str | None = None
    code: str | None = None
    context: dict[str, Any] | None = None


class IngestionSummary(BaseModel):
    """
    End-of-run ingestion summary.
    """

    model_config = ConfigDict(frozen=True)

    rows_processed: int
    rows_failed: int
    validation_errors: list[RowValidationError] = Field(default_factory=list)
    pipeline_status: str = "failed"
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    inferred_category: str | None = None
    category_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    category_inference_status: str | None = None
    category_inference_evidence: list[str] = Field(default_factory=list)
    category_alternatives: list[dict[str, Any]] = Field(default_factory=list)
