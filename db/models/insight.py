"""
db/models/insight.py

Insight model storing structured reasoning outputs.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from db.models.analytical_metric import AnalyticalMetric
    from db.models.client import Client
    from db.models.dataset import Dataset


class Insight(Base, TimestampMixin):
    __tablename__ = "insights"

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
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="SET NULL"),
        nullable=True,
    )
    metric_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analytical_metrics.id", ondelete="SET NULL"),
        nullable=True,
    )

    insight_type: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    output_payload: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        comment="Structured JSON output from the reasoning layer",
    )
    evidence_payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    priority: Mapped[str | None] = mapped_column(String(20), nullable=True)
    confidence_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    client: Mapped["Client"] = relationship("Client", back_populates="insights")
    dataset: Mapped["Dataset | None"] = relationship("Dataset", back_populates="insights")
    metric: Mapped["AnalyticalMetric | None"] = relationship("AnalyticalMetric", back_populates="insights")

    __table_args__ = (
        Index("ix_insights_client_generated_at", "client_id", "generated_at"),
        Index("ix_insights_client_type_generated_at", "client_id", "insight_type", "generated_at"),
        Index("ix_insights_dataset_generated_at", "dataset_id", "generated_at"),
        Index("ix_insights_metric_generated_at", "metric_id", "generated_at"),
    )
