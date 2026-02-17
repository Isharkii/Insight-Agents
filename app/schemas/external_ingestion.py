"""
app/schemas/external_ingestion.py

Response schemas for external ingestion operations.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExternalIngestionSummaryResponse(BaseModel):
    """
    API response model for one source ingestion summary.
    """

    source: str
    records_inserted: int = Field(..., ge=0)
    failed_records: int = Field(..., ge=0)
