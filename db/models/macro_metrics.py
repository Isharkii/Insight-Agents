"""
db/models/macro_metrics.py

SQLAlchemy models for macroeconomic time-series storage.
PostgreSQL-oriented, UUID primary keys, and indexed for range queries.
"""

from __future__ import annotations

import uuid
from datetime import datetime, date
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base


class MacroMetricRun(Base):
    """
    Version/run header for macroeconomic ingestion snapshots.
    """

    __tablename__ = "macro_metric_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    source_key: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="Data source key: world_bank, fred, imf, etc.",
    )
    country_code: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
    )
    run_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    source_release_ts: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    is_current: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    metrics: Mapped[list["MacroMetric"]] = relationship(
        "MacroMetric",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "source_key",
            "country_code",
            "run_version",
            name="uq_macro_metric_runs_source_country_version",
        ),
        Index("ix_macro_metric_runs_country_code", "country_code"),
        Index("ix_macro_metric_runs_source_key", "source_key"),
        Index(
            "ix_macro_metric_runs_source_country_current",
            "source_key",
            "country_code",
            "is_current",
        ),
    )


class MacroMetric(Base):
    """
    Fact table for macroeconomic time-series observations.
    """

    __tablename__ = "macro_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("macro_metric_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    country_code: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
    )
    metric_name: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    frequency: Mapped[str] = mapped_column(
        String(1),
        nullable=False,
        comment="M=monthly, Q=quarterly",
    )
    period_start: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )
    period_end: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )
    value: Mapped[float] = mapped_column(
        Numeric(20, 6),
        nullable=False,
    )
    unit: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
    )
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    run: Mapped["MacroMetricRun"] = relationship(
        "MacroMetricRun",
        back_populates="metrics",
    )

    __table_args__ = (
        CheckConstraint(
            "frequency IN ('M', 'Q')",
            name="ck_macro_metrics_frequency",
        ),
        UniqueConstraint(
            "run_id",
            "country_code",
            "metric_name",
            "frequency",
            "period_end",
            name="uq_macro_metrics_run_country_metric_frequency_period_end",
        ),
        Index("ix_macro_metrics_country_code", "country_code"),
        Index("ix_macro_metrics_metric_name", "metric_name"),
        Index("ix_macro_metrics_period_end", "period_end"),
        Index(
            "ix_macro_metrics_country_metric_frequency_period_end",
            "country_code",
            "metric_name",
            "frequency",
            "period_end",
        ),
        Index("ix_macro_metrics_run_id", "run_id"),
    )
