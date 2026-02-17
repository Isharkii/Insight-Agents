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


class CSVIngestionSummaryResponse(BaseModel):
    """
    API response model for CSV ingestion summary.
    """

    rows_processed: int = Field(..., ge=0)
    rows_failed: int = Field(..., ge=0)
    validation_errors: list[CSVValidationErrorResponse] = Field(default_factory=list)
