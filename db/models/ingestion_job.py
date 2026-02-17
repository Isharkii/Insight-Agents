"""
db/models/ingestion_job.py

Ingestion job model for asynchronous orchestration tracking.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base, TimestampMixin


class IngestionJobType:
    CSV = "csv"
    API = "api"
    COMPETITOR_SCRAPING = "competitor_scraping"


class IngestionJobStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestionJob(Base, TimestampMixin):
    __tablename__ = "ingestion_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    job_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="csv, api, competitor_scraping",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=IngestionJobStatus.PENDING,
    )
    request_payload: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Submitted request parameters and metadata",
    )
    result_payload: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Execution result metadata",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    __table_args__ = (
        Index("ix_ingestion_jobs_job_type", "job_type"),
        Index("ix_ingestion_jobs_status", "status"),
        Index("ix_ingestion_jobs_created_at", "created_at"),
        Index("ix_ingestion_jobs_job_type_status", "job_type", "status"),
    )
