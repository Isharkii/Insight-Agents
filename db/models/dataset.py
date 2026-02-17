"""
db/models/dataset.py

Dataset model representing one uploaded or ingested dataset.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from db.models.analytical_metric import AnalyticalMetric
    from db.models.client import Client
    from db.models.insight import Insight


class DatasetStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    client_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("clients.id", ondelete="CASCADE"),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="csv, excel, api, manual",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=DatasetStatus.PENDING,
    )

    file_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)

    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    schema_meta: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    file_meta: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Additional uploaded-file metadata",
    )

    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    client: Mapped["Client"] = relationship("Client", back_populates="datasets")
    analytical_metrics: Mapped[list["AnalyticalMetric"]] = relationship(
        "AnalyticalMetric",
        back_populates="dataset",
        passive_deletes=True,
    )
    insights: Mapped[list["Insight"]] = relationship(
        "Insight",
        back_populates="dataset",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_datasets_client_id", "client_id"),
        Index("ix_datasets_status", "status"),
        Index("ix_datasets_source_type", "source_type"),
        Index("ix_datasets_processed_at", "processed_at"),
        Index("ix_datasets_client_status", "client_id", "status"),
    )
