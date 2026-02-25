"""
app/schemas/csv_ingestion.py

Response schemas for CSV ingestion endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CSVValidationErrorResponse(BaseModel):
    """
    API response model for one row-level validation error.
    """

    row_number: int = Field(..., ge=1)
    message: str
    column: str | None = None
    value: str | None = None
    code: str | None = None
    context: dict[str, object] | None = None


class CSVIngestionSummaryResponse(BaseModel):
    """
    API response model for CSV ingestion summary.
    """

    rows_processed: int = Field(..., ge=0)
    rows_failed: int = Field(..., ge=0)
    pipeline_status: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    provenance: dict[str, object] = Field(default_factory=dict)
    diagnostics: dict[str, object] = Field(default_factory=dict)
    validation_errors: list[CSVValidationErrorResponse] = Field(default_factory=list)
    inferred_category: str | None = None
    category_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    category_inference_status: str | None = None
