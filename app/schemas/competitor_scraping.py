"""
app/schemas/competitor_scraping.py

Response schemas for competitor scraping operations.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CompetitorScrapeSummaryResponse(BaseModel):
    """
    API response model for one competitor scrape summary.
    """

    competitor: str
    records_scraped: int = Field(..., ge=0)
    records_inserted: int = Field(..., ge=0)
    failed_pages: int = Field(..., ge=0)
    status: str
    errors: list[str] = Field(default_factory=list)
