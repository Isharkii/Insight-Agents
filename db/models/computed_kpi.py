"""
db/models/computed_kpi.py

Persisted output of the KPI calculation engine.
One row per entity per period window.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base

_UPSERT_CONSTRAINT = "uq_computed_kpis_tenant_entity_period"


class ComputedKPI(Base):
    """
    Stores the structured KPI results produced by KPIService for a given
    entity and time window.

    ``computed_kpis`` holds the full JSONB payload, e.g.::

        {
            "mrr":         {"value": 12000.0, "unit": "currency"},
            "churn_rate":  {"value": 0.05,    "unit": "rate"},
            "ltv":         {"value": 2400.0,  "unit": "currency"},
            "growth_rate": {"value": 0.12,    "unit": "rate"}
        }

    The unique constraint on ``(tenant_id, entity_id, period_start, period_end)``
    drives upsert semantics: re-running the engine for the same window
    updates the existing row instead of inserting a duplicate.
    """

    __tablename__ = "computed_kpis"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    tenant_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        server_default="legacy",
        comment="Owning tenant identifier.",
    )
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenant_entities.id", ondelete="RESTRICT"),
        nullable=False,
        comment="Stable tenant-local entity identifier.",
    )
    entity_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Denormalized display label retained for backward compatibility.",
    )
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Inclusive start of the measurement period (UTC)",
    )
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Inclusive end of the measurement period (UTC)",
    )
    computed_kpis: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment="Structured KPI results keyed by metric name",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    analytics_version: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Pipeline version that produced this row; NULL = pre-versioning",
    )
    dataset_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="SHA-256 hex digest of canonical input rows at computation time",
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "entity_id",
            "period_start",
            "period_end",
            name=_UPSERT_CONSTRAINT,
        ),
        Index(
            "ix_computed_kpis_tenant_entity_period_start",
            "tenant_id",
            "entity_id",
            "period_start",
        ),
        Index(
            "ix_computed_kpis_tenant_entity_name_period_start",
            "tenant_id",
            "entity_name",
            "period_start",
        ),
        Index(
            "ix_computed_kpis_tenant_period_start",
            "tenant_id",
            "period_start",
        ),
        Index("ix_computed_kpis_entity_name", "entity_name"),
        Index("ix_computed_kpis_period_start", "period_start"),
    )
