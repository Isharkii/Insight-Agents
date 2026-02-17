"""
Schemas for ingestion orchestration trigger and status endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class IngestionJobAcceptedResponse(BaseModel):
    job_id: UUID
    job_type: str
    status: str
    created_at: datetime


class IngestionJobStatusResponse(BaseModel):
    job_id: UUID
    job_type: str
    status: str
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    request_payload: dict[str, Any] | None = None
    result_payload: dict[str, Any] | None = None
    error_message: str | None = None


class IngestionStatusListResponse(BaseModel):
    jobs: list[IngestionJobStatusResponse] = Field(default_factory=list)
