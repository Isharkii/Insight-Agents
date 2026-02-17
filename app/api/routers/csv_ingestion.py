"""
app/api/routers/csv_ingestion.py

CSV ingestion HTTP endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_csv_upload
from app.schemas.csv_ingestion import CSVIngestionSummaryResponse, CSVValidationErrorResponse
from app.services.csv_ingestion_service import (
    CSVHeaderValidationError,
    CSVPersistenceError,
    CSVIngestionService,
    get_csv_ingestion_service,
)
from db.session import get_db

router = APIRouter(tags=["ingestion"])


@router.post("/upload-csv", response_model=CSVIngestionSummaryResponse)
def upload_csv(
    file: UploadFile = Depends(get_csv_upload),
    db: Session = Depends(get_db),
    ingestion_service: CSVIngestionService = Depends(get_csv_ingestion_service),
) -> CSVIngestionSummaryResponse:
    """
    Ingest one CSV file into canonical insight records.
    """

    try:
        summary = ingestion_service.ingest_csv(upload_file=file, db=db)
    except CSVHeaderValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except CSVPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to persist valid CSV rows.",
        ) from exc
    finally:
        file.file.close()

    return CSVIngestionSummaryResponse(
        rows_processed=summary.rows_processed,
        rows_failed=summary.rows_failed,
        validation_errors=[
            CSVValidationErrorResponse(
                row_number=error.row_number,
                column=error.column,
                message=error.message,
                value=error.value,
            )
            for error in summary.validation_errors
        ],
    )
