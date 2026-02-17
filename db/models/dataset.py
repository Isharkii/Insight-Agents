"""
db/models/dataset.py

Dataset model — represents a single uploaded or ingested file of business data.
Each dataset belongs to one client and drives one or more analytical passes.
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from db.models.analytical_metric import AnalyticalMetric
    from db.models.client import Client
    from db.models.insight import Insight


class DatasetStatus:
    """Valid status transitions for a dataset."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Dataset(Base, TimestampMixin):
    """
    Represents one uploaded or ingested file of business data.

    schema_meta stores the detected column structure so downstream
    layers can reason about available fields without re-reading the file.

    file_meta stores raw upload metadata (original filename, size, MIME type, etc.)
    to support auditability and future re-ingestion.
    """

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

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Human-readable label for this dataset",
    )

    source_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Origin format: csv, excel, api, manual",
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=DatasetStatus.PENDING,
        comment="Pipeline state: pending → processing → ready | failed",
    )

    row_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Populated after successful ingestion",
    )

    schema_meta: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Detected column names and inferred data types",
    )

    file_meta: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Upload metadata: filename, size_bytes, mime_type, checksum",
    )

    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when processing completed or failed",
    )

    # ── Relationships ──────────────────────────────────────────────────────────

    client: Mapped["Client"] = relationship(
        "Client",
        back_populates="datasets",
    )

    analytical_metrics: Mapped[list["AnalyticalMetric"]] = relationship(
        "AnalyticalMetric",
        back_populates="dataset",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    insights: Mapped[list["Insight"]] = relationship(
        "Insight",
        back_populates="dataset",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    # ── Indexes ────────────────────────────────────────────────────────────────

    __table_args__ = (
        Index("ix_datasets_client_id", "client_id"),
        Index("ix_datasets_status", "status"),
        Index("ix_datasets_source_type", "source_type"),
        Index("ix_datasets_client_status", "client_id", "status"),
    )

    def __repr__(self) -> str:
        return (
            f"<Dataset id={self.id} name={self.name!r} "
            f"client_id={self.client_id} status={self.status!r}>"
        )
