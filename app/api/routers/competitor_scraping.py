"""
app/api/routers/competitor_scraping.py

Competitor scraping ingestion endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.schemas.competitor_scraping import CompetitorScrapeSummaryResponse
from app.services.competitor_scraping_service import (
    CompetitorScrapingService,
    get_competitor_scraping_service,
)
from db.session import get_db

router = APIRouter(tags=["competitor-scraping"])


@router.post("/ingest-competitors", response_model=list[CompetitorScrapeSummaryResponse])
def ingest_competitors(
    competitor: str | None = Query(default=None, description="Optional competitor name filter"),
    db: Session = Depends(get_db),
    scraping_service: CompetitorScrapingService = Depends(get_competitor_scraping_service),
) -> list[CompetitorScrapeSummaryResponse]:
    """
    Run competitor scraping for all configured competitors or one selected competitor.
    """

    try:
        summaries = scraping_service.ingest(db=db, competitor=competitor)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return [
        CompetitorScrapeSummaryResponse(
            competitor=summary.competitor,
            records_scraped=summary.records_scraped,
            records_inserted=summary.records_inserted,
            failed_pages=summary.failed_pages,
            status=summary.status,
            errors=summary.errors,
        )
        for summary in summaries
    ]
