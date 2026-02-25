"""
db/models/benchmarks.py

SQLAlchemy models for benchmark storage.
PostgreSQL-compatible schema with UUID keys, JSONB support, and
time-series snapshot storage.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base, TimestampMixin


class IndustryCategory(Base, TimestampMixin):
    """
    Optional industry taxonomy table used to group benchmarks.
    """

    __tablename__ = "industry_categories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    industry_key: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Stable industry identifier (e.g. saas, retail, healthcare).",
    )
    industry_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Display label for the industry category.",
    )
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    benchmarks: Mapped[list["Benchmark"]] = relationship(
        "Benchmark",
        back_populates="industry_category",
    )

    __table_args__ = (
        UniqueConstraint("industry_key", name="uq_industry_categories_industry_key"),
        Index("ix_industry_categories_industry_key", "industry_key"),
        Index("ix_industry_categories_industry_name", "industry_name"),
    )


class Benchmark(Base, TimestampMixin):
    """
    Benchmark definition/header.
    """

    __tablename__ = "benchmarks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    benchmark_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Human-readable benchmark name.",
    )
    industry: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
        comment="Industry bucket used for filtering and query performance.",
    )
    industry_category_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("industry_categories.id", ondelete="SET NULL"),
        nullable=True,
    )
    source: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="Primary source system/provider for this benchmark.",
    )
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    industry_category: Mapped["IndustryCategory | None"] = relationship(
        "IndustryCategory",
        back_populates="benchmarks",
    )
    metrics: Mapped[list["BenchmarkMetric"]] = relationship(
        "BenchmarkMetric",
        back_populates="benchmark",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "benchmark_name",
            "industry",
            name="uq_benchmarks_name_industry",
        ),
        Index("ix_benchmarks_industry", "industry"),
        Index("ix_benchmarks_benchmark_name", "benchmark_name"),
        Index("ix_benchmarks_industry_category_id", "industry_category_id"),
        Index("ix_benchmarks_is_active", "is_active"),
    )


class BenchmarkMetric(Base, TimestampMixin):
    """
    Metric catalog for a benchmark.
    """

    __tablename__ = "benchmark_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    benchmark_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("benchmarks.id", ondelete="CASCADE"),
        nullable=False,
    )
    metric_name: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
    )
    unit: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
    )
    metric_config_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Flexible metric definition/options payload.",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    benchmark: Mapped["Benchmark"] = relationship(
        "Benchmark",
        back_populates="metrics",
    )
    snapshots: Mapped[list["BenchmarkSnapshot"]] = relationship(
        "BenchmarkSnapshot",
        back_populates="benchmark_metric",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "benchmark_id",
            "metric_name",
            name="uq_benchmark_metrics_benchmark_metric_name",
        ),
        Index("ix_benchmark_metrics_metric_name", "metric_name"),
        Index("ix_benchmark_metrics_benchmark_id", "benchmark_id"),
        Index("ix_benchmark_metrics_is_active", "is_active"),
    )


class BenchmarkSnapshot(Base, TimestampMixin):
    """
    Time-series fact table for benchmark metric values.
    """

    __tablename__ = "benchmark_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    benchmark_metric_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("benchmark_metrics.id", ondelete="CASCADE"),
        nullable=False,
    )
    period_start: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )
    period_end: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )
    frequency: Mapped[str] = mapped_column(
        String(1),
        nullable=False,
        comment="M=monthly, Q=quarterly, Y=yearly",
    )
    snapshot_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Version of the observation for the same period (for revisions).",
    )
    metric_value: Mapped[float | None] = mapped_column(
        Numeric(20, 6),
        nullable=True,
    )
    metric_payload_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Optional flexible metric payload for non-scalar values.",
    )
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    benchmark_metric: Mapped["BenchmarkMetric"] = relationship(
        "BenchmarkMetric",
        back_populates="snapshots",
    )

    __table_args__ = (
        CheckConstraint(
            "frequency IN ('M', 'Q', 'Y')",
            name="ck_benchmark_snapshots_frequency",
        ),
        CheckConstraint(
            "metric_value IS NOT NULL OR metric_payload_json IS NOT NULL",
            name="ck_benchmark_snapshots_value_or_payload",
        ),
        UniqueConstraint(
            "benchmark_metric_id",
            "period_end",
            "snapshot_version",
            name="uq_benchmark_snapshots_metric_period_version",
        ),
        Index("ix_benchmark_snapshots_period_end", "period_end"),
        Index(
            "ix_benchmark_snapshots_metric_period_end",
            "benchmark_metric_id",
            "period_end",
        ),
        Index(
            "ix_benchmark_snapshots_period_frequency",
            "period_end",
            "frequency",
        ),
    )
