"""
Async ingestion orchestration endpoints.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_csv_upload
from app.schemas.ingestion_orchestrator import (
    IngestionJobAcceptedResponse,
    IngestionJobStatusResponse,
    IngestionStatusListResponse,
)
from app.services.ingestion_orchestrator_service import (
    FastAPIBackgroundTaskExecutor,
    IngestionOrchestratorService,
    get_ingestion_orchestrator_service,
)
from db.models.ingestion_job import IngestionJob
from db.session import get_db

router = APIRouter(tags=["ingestion-orchestrator"])


@router.post(
    "/ingestion/csv",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestionJobAcceptedResponse,
)
def trigger_csv_ingestion(
    background_tasks: BackgroundTasks,
    file: UploadFile = Depends(get_csv_upload),
    client_name: str | None = Query(default=None, description="Optional client name to scope mapping config"),
    mapping_config_name: str | None = Query(default=None, description="Optional explicit mapping config name"),
    db: Session = Depends(get_db),
    orchestrator: IngestionOrchestratorService = Depends(get_ingestion_orchestrator_service),
) -> IngestionJobAcceptedResponse:
    try:
        job = orchestrator.trigger_csv_ingestion(
            db=db,
            executor=FastAPIBackgroundTaskExecutor(background_tasks),
            upload_file=file,
            client_name=client_name,
            mapping_config_name=mapping_config_name,
        )
    finally:
        file.file.close()

    return IngestionJobAcceptedResponse(
        job_id=job.id,
        job_type=job.job_type,
        status=job.status,
        created_at=job.created_at,
    )


@router.post(
    "/ingestion/api",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestionJobAcceptedResponse,
)
def trigger_api_ingestion(
    background_tasks: BackgroundTasks,
    source: str | None = Query(default=None, description="Optional external source filter"),
    db: Session = Depends(get_db),
    orchestrator: IngestionOrchestratorService = Depends(get_ingestion_orchestrator_service),
) -> IngestionJobAcceptedResponse:
    job = orchestrator.trigger_api_ingestion(
        db=db,
        executor=FastAPIBackgroundTaskExecutor(background_tasks),
        source=source,
    )
    return IngestionJobAcceptedResponse(
        job_id=job.id,
        job_type=job.job_type,
        status=job.status,
        created_at=job.created_at,
    )


@router.post(
    "/ingestion/competitor-scraping",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestionJobAcceptedResponse,
)
def trigger_competitor_scraping(
    background_tasks: BackgroundTasks,
    competitor: str | None = Query(default=None, description="Optional competitor name filter"),
    db: Session = Depends(get_db),
    orchestrator: IngestionOrchestratorService = Depends(get_ingestion_orchestrator_service),
) -> IngestionJobAcceptedResponse:
    job = orchestrator.trigger_competitor_scraping(
        db=db,
        executor=FastAPIBackgroundTaskExecutor(background_tasks),
        competitor=competitor,
    )
    return IngestionJobAcceptedResponse(
        job_id=job.id,
        job_type=job.job_type,
        status=job.status,
        created_at=job.created_at,
    )


@router.get("/ingestion-status", response_model=IngestionStatusListResponse)
def get_ingestion_status(
    job_id: UUID | None = Query(default=None, description="Optional ingestion job ID"),
    job_type: str | None = Query(default=None, description="Optional job type filter"),
    status_filter: str | None = Query(default=None, alias="status", description="Optional status filter"),
    limit: int = Query(default=100, ge=1, le=500, description="Max jobs returned when listing"),
    db: Session = Depends(get_db),
    orchestrator: IngestionOrchestratorService = Depends(get_ingestion_orchestrator_service),
) -> IngestionStatusListResponse:
    if job_id is not None:
        job = orchestrator.get_job_status(db=db, job_id=job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ingestion job not found: {job_id}",
            )
        return IngestionStatusListResponse(jobs=[_to_status_response(job)])

    jobs = orchestrator.list_job_statuses(
        db=db,
        limit=limit,
        job_type=job_type,
        status=status_filter,
    )
    return IngestionStatusListResponse(jobs=[_to_status_response(job) for job in jobs])


def _to_status_response(job: IngestionJob) -> IngestionJobStatusResponse:
    return IngestionJobStatusResponse(
        job_id=job.id,
        job_type=job.job_type,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        request_payload=job.request_payload,
        result_payload=job.result_payload,
        error_message=job.error_message,
    )
