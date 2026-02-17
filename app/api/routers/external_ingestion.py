"""
app/api/routers/external_ingestion.py

External data ingestion HTTP endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.schemas.external_ingestion import ExternalIngestionSummaryResponse
from app.services.external_ingestion_service import (
    ExternalIngestionService,
    get_external_ingestion_service,
)
from db.session import get_db

router = APIRouter(tags=["external-ingestion"])


@router.post("/ingest-external", response_model=list[ExternalIngestionSummaryResponse])
def ingest_external(
    source: str | None = Query(default=None, description="Optional source filter"),
    db: Session = Depends(get_db),
    ingestion_service: ExternalIngestionService = Depends(get_external_ingestion_service),
) -> list[ExternalIngestionSummaryResponse]:
    """
    Run ingestion for all configured external sources or one selected source.
    """

    try:
        summaries = ingestion_service.ingest(db=db, source=source)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return [
        ExternalIngestionSummaryResponse(
            source=summary.source,
            records_inserted=summary.records_inserted,
            failed_records=summary.failed_records,
        )
        for summary in summaries
    ]
