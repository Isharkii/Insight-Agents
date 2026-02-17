"""
Repository for ingestion job lifecycle persistence and status lookup.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from db.models.ingestion_job import IngestionJob, IngestionJobStatus


class IngestionJobRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create_job(
        self,
        *,
        job_type: str,
        request_payload: dict[str, Any] | None = None,
    ) -> IngestionJob:
        job = IngestionJob(
            job_type=job_type,
            status=IngestionJobStatus.PENDING,
            request_payload=request_payload,
        )
        self._session.add(job)
        self._session.flush()
        self._session.refresh(job)
        return job

    def get_job(self, job_id: uuid.UUID) -> IngestionJob | None:
        return self._session.get(IngestionJob, job_id)

    def list_jobs(
        self,
        *,
        limit: int = 100,
        job_type: str | None = None,
        status: str | None = None,
    ) -> list[IngestionJob]:
        stmt: Select[tuple[IngestionJob]] = select(IngestionJob)

        if job_type:
            stmt = stmt.where(IngestionJob.job_type == job_type)
        if status:
            stmt = stmt.where(IngestionJob.status == status)

        stmt = stmt.order_by(IngestionJob.created_at.desc()).limit(max(1, limit))
        return list(self._session.scalars(stmt).all())

    def mark_running(self, *, job_id: uuid.UUID) -> IngestionJob | None:
        job = self.get_job(job_id)
        if job is None:
            return None
        job.status = IngestionJobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        job.completed_at = None
        job.error_message = None
        return job

    def mark_completed(
        self,
        *,
        job_id: uuid.UUID,
        result_payload: dict[str, Any] | None = None,
    ) -> IngestionJob | None:
        job = self.get_job(job_id)
        if job is None:
            return None
        job.status = IngestionJobStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc)
        job.result_payload = result_payload
        job.error_message = None
        return job

    def mark_failed(
        self,
        *,
        job_id: uuid.UUID,
        error_message: str,
        result_payload: dict[str, Any] | None = None,
    ) -> IngestionJob | None:
        job = self.get_job(job_id)
        if job is None:
            return None
        job.status = IngestionJobStatus.FAILED
        job.completed_at = datetime.now(timezone.utc)
        job.error_message = error_message
        if result_payload is not None:
            job.result_payload = result_payload
        return job
