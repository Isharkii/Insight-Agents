"""
db/models/analytical_metric.py

AnalyticalMetric model for KPI, trend, and anomaly metric records.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import Date, DateTime, ForeignKey, Index, Numeric, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from db.models.client import Client
    from db.models.dataset import Dataset
    from db.models.insight import Insight


class AnalyticalMetricType:
    KPI = "kpi"
    TREND = "trend"
    ANOMALY = "anomaly"


class AnalyticalMetric(Base, TimestampMixin):
    __tablename__ = "analytical_metrics"

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

    metric_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=AnalyticalMetricType.KPI,
        comment="kpi, trend, anomaly",
    )
    metric_key: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
        comment="Stable metric identifier",
    )
    metric_label: Mapped[str | None] = mapped_column(String(255), nullable=True)

    metric_value: Mapped[Decimal | None] = mapped_column(Numeric(20, 6), nullable=True)
    unit: Mapped[str | None] = mapped_column(String(40), nullable=True)

    period_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    period_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    measured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    dimensions: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Dynamic grouping dimensions (region, channel, product)",
    )
    metric_payload: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Flexible structured metric data",
    )

    source: Mapped[str | None] = mapped_column(String(80), nullable=True)
    confidence_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)

    client: Mapped["Client"] = relationship("Client", back_populates="analytical_metrics")
    dataset: Mapped["Dataset | None"] = relationship("Dataset", back_populates="analytical_metrics")
    insights: Mapped[list["Insight"]] = relationship(
        "Insight",
        back_populates="metric",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_analytical_metrics_client_metric_measured", "client_id", "metric_key", "measured_at"),
        Index("ix_analytical_metrics_client_type_measured", "client_id", "metric_type", "measured_at"),
        Index("ix_analytical_metrics_dataset_measured", "dataset_id", "measured_at"),
        Index("ix_analytical_metrics_client_metric_period", "client_id", "metric_key", "period_start"),
    )
