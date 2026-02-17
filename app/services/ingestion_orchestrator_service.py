"""
Orchestrator service for async ingestion job dispatch and lifecycle tracking.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Protocol

from fastapi import BackgroundTasks, UploadFile
from sqlalchemy.orm import Session, sessionmaker

from app.services.competitor_scraping_service import (
    CompetitorScrapingService,
    get_competitor_scraping_service,
)
from app.services.csv_ingestion_service import CSVIngestionService, get_csv_ingestion_service
from app.services.external_ingestion_service import (
    ExternalIngestionService,
    get_external_ingestion_service,
)
from db.models.ingestion_job import IngestionJob, IngestionJobType
from db.repositories.ingestion_job_repository import IngestionJobRepository

logger = logging.getLogger(__name__)


class IngestionTaskExecutor(Protocol):
    def submit(self, task: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        ...


class FastAPIBackgroundTaskExecutor:
    def __init__(self, background_tasks: BackgroundTasks) -> None:
        self._background_tasks = background_tasks

    def submit(self, task: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        self._background_tasks.add_task(task, *args, **kwargs)


class IngestionOrchestratorService:
    """
    Coordinates job creation, background execution, and status persistence.
    """

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session] | None = None,
        csv_service: CSVIngestionService | None = None,
        external_service: ExternalIngestionService | None = None,
        competitor_service: CompetitorScrapingService | None = None,
    ) -> None:
        if session_factory is None:
            from db.session import SessionLocal

            self._session_factory = SessionLocal
        else:
            self._session_factory = session_factory

        self._csv_service = csv_service or get_csv_ingestion_service()
        self._external_service = external_service or get_external_ingestion_service()
        self._competitor_service = competitor_service or get_competitor_scraping_service()

    def trigger_csv_ingestion(
        self,
        *,
        db: Session,
        executor: IngestionTaskExecutor,
        upload_file: UploadFile,
        client_name: str | None = None,
        mapping_config_name: str | None = None,
    ) -> IngestionJob:
        temp_file_path, file_size = self._persist_temp_upload(upload_file)
        file_name = upload_file.filename or "upload.csv"
        request_payload = {
            "file_name": file_name,
            "content_type": upload_file.content_type,
            "file_size_bytes": file_size,
            "client_name": client_name,
            "mapping_config_name": mapping_config_name,
        }

        repository = IngestionJobRepository(db)
        with db.begin():
            job = repository.create_job(
                job_type=IngestionJobType.CSV,
                request_payload=request_payload,
            )

        try:
            executor.submit(
                self._run_csv_ingestion_job,
                job.id,
                temp_file_path,
                file_name,
                client_name,
                mapping_config_name,
            )
        except Exception:
            self._delete_file_quietly(temp_file_path)
            with db.begin():
                repository.mark_failed(
                    job_id=job.id,
                    error_message="Failed to schedule CSV ingestion job.",
                )
            raise

        return job

    def trigger_api_ingestion(
        self,
        *,
        db: Session,
        executor: IngestionTaskExecutor,
        source: str | None = None,
    ) -> IngestionJob:
        repository = IngestionJobRepository(db)
        with db.begin():
            job = repository.create_job(
                job_type=IngestionJobType.API,
                request_payload={"source": source},
            )

        try:
            executor.submit(self._run_api_ingestion_job, job.id, source)
        except Exception:
            with db.begin():
                repository.mark_failed(
                    job_id=job.id,
                    error_message="Failed to schedule API ingestion job.",
                )
            raise

        return job

    def trigger_competitor_scraping(
        self,
        *,
        db: Session,
        executor: IngestionTaskExecutor,
        competitor: str | None = None,
    ) -> IngestionJob:
        repository = IngestionJobRepository(db)
        with db.begin():
            job = repository.create_job(
                job_type=IngestionJobType.COMPETITOR_SCRAPING,
                request_payload={"competitor": competitor},
            )

        try:
            executor.submit(self._run_competitor_scraping_job, job.id, competitor)
        except Exception:
            with db.begin():
                repository.mark_failed(
                    job_id=job.id,
                    error_message="Failed to schedule competitor scraping job.",
                )
            raise

        return job

    def get_job_status(self, *, db: Session, job_id: uuid.UUID) -> IngestionJob | None:
        repository = IngestionJobRepository(db)
        return repository.get_job(job_id)

    def list_job_statuses(
        self,
        *,
        db: Session,
        limit: int = 100,
        job_type: str | None = None,
        status: str | None = None,
    ) -> list[IngestionJob]:
        repository = IngestionJobRepository(db)
        return repository.list_jobs(
            limit=limit,
            job_type=job_type,
            status=status,
        )

    def _run_csv_ingestion_job(
        self,
        job_id: uuid.UUID,
        temp_file_path: str,
        file_name: str,
        client_name: str | None,
        mapping_config_name: str | None,
    ) -> None:
        with self._session_factory() as db:
            repository = IngestionJobRepository(db)
            try:
                running_job = repository.mark_running(job_id=job_id)
                if running_job is None:
                    raise RuntimeError(f"Ingestion job not found: {job_id}")
                db.commit()

                with open(temp_file_path, "rb") as file_handle:
                    upload_file = UploadFile(file=file_handle, filename=file_name)
                    summary = self._csv_service.ingest_csv(
                        upload_file=upload_file,
                        db=db,
                        client_name=client_name,
                        mapping_config_name=mapping_config_name,
                    )

                completed_job = repository.mark_completed(
                    job_id=job_id,
                    result_payload={
                        "rows_processed": summary.rows_processed,
                        "rows_failed": summary.rows_failed,
                        "validation_error_count": len(summary.validation_errors),
                        "validation_errors": [
                            {
                                "row_number": error.row_number,
                                "column": error.column,
                                "message": error.message,
                                "value": error.value,
                            }
                            for error in summary.validation_errors[:100]
                        ],
                    },
                )
                if completed_job is None:
                    raise RuntimeError(f"Ingestion job not found: {job_id}")
                db.commit()
            except Exception as exc:
                self._mark_job_failed(db=db, job_id=job_id, exc=exc)
            finally:
                self._delete_file_quietly(temp_file_path)

    def _run_api_ingestion_job(self, job_id: uuid.UUID, source: str | None) -> None:
        with self._session_factory() as db:
            repository = IngestionJobRepository(db)
            try:
                running_job = repository.mark_running(job_id=job_id)
                if running_job is None:
                    raise RuntimeError(f"Ingestion job not found: {job_id}")
                db.commit()

                summaries = self._external_service.ingest(db=db, source=source)
                result_payload = {
                    "source_summaries": [
                        {
                            "source": summary.source,
                            "records_inserted": summary.records_inserted,
                            "failed_records": summary.failed_records,
                        }
                        for summary in summaries
                    ],
                    "records_inserted_total": sum(summary.records_inserted for summary in summaries),
                    "failed_records_total": sum(summary.failed_records for summary in summaries),
                }
                completed_job = repository.mark_completed(job_id=job_id, result_payload=result_payload)
                if completed_job is None:
                    raise RuntimeError(f"Ingestion job not found: {job_id}")
                db.commit()
            except Exception as exc:
                self._mark_job_failed(db=db, job_id=job_id, exc=exc)

    def _run_competitor_scraping_job(self, job_id: uuid.UUID, competitor: str | None) -> None:
        with self._session_factory() as db:
            repository = IngestionJobRepository(db)
            try:
                running_job = repository.mark_running(job_id=job_id)
                if running_job is None:
                    raise RuntimeError(f"Ingestion job not found: {job_id}")
                db.commit()

                summaries = self._competitor_service.ingest(db=db, competitor=competitor)
                result_payload = {
                    "competitor_summaries": [
                        {
                            "competitor": summary.competitor,
                            "records_scraped": summary.records_scraped,
                            "records_inserted": summary.records_inserted,
                            "failed_pages": summary.failed_pages,
                            "status": summary.status,
                            "errors": summary.errors,
                        }
                        for summary in summaries
                    ],
                    "records_inserted_total": sum(summary.records_inserted for summary in summaries),
                    "failed_pages_total": sum(summary.failed_pages for summary in summaries),
                }
                completed_job = repository.mark_completed(job_id=job_id, result_payload=result_payload)
                if completed_job is None:
                    raise RuntimeError(f"Ingestion job not found: {job_id}")
                db.commit()
            except Exception as exc:
                self._mark_job_failed(db=db, job_id=job_id, exc=exc)

    def _mark_job_failed(self, *, db: Session, job_id: uuid.UUID, exc: Exception) -> None:
        repository = IngestionJobRepository(db)
        error_message = f"{type(exc).__name__}: {exc}"
        logger.exception("Ingestion job failed id=%s error=%s", job_id, error_message)
        try:
            db.rollback()
            failed_job = repository.mark_failed(
                job_id=job_id,
                error_message=error_message[:2000],
            )
            if failed_job is None:
                logger.error("Unable to mark ingestion job as failed because it was not found id=%s", job_id)
            db.commit()
        except Exception:
            db.rollback()
            logger.exception("Failed to persist failed ingestion job state id=%s", job_id)

    def _persist_temp_upload(self, upload_file: UploadFile) -> tuple[str, int]:
        file_name = upload_file.filename or "upload.csv"
        _, ext = os.path.splitext(file_name)
        suffix = ext if ext else ".csv"
        upload_file.file.seek(0)

        with tempfile.NamedTemporaryFile(delete=False, prefix="ingestion_job_", suffix=suffix) as temp_file:
            while True:
                chunk = upload_file.file.read(1024 * 1024)
                if not chunk:
                    break
                temp_file.write(chunk)
            temp_path = temp_file.name
            file_size = temp_file.tell()

        upload_file.file.seek(0)
        return temp_path, file_size

    def _delete_file_quietly(self, file_path: str) -> None:
        try:
            os.remove(file_path)
        except OSError:
            return


@lru_cache(maxsize=1)
def get_ingestion_orchestrator_service() -> IngestionOrchestratorService:
    return IngestionOrchestratorService()
