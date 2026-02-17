"""
db/models/computed_kpi.py

Persisted output of the KPI calculation engine.
One row per entity per period window.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base

_UPSERT_CONSTRAINT = "uq_computed_kpis_entity_period"


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

    The unique constraint on ``(entity_name, period_start, period_end)``
    drives upsert semantics: re-running the engine for the same window
    updates the existing row instead of inserting a duplicate.
    """

    __tablename__ = "computed_kpis"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    entity_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Client or competitor entity name; matches CanonicalInsightRecord.entity_name",
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

    __table_args__ = (
        UniqueConstraint(
            "entity_name",
            "period_start",
            "period_end",
            name=_UPSERT_CONSTRAINT,
        ),
        Index("ix_computed_kpis_entity_name", "entity_name"),
        Index("ix_computed_kpis_period_start", "period_start"),
        Index(
            "ix_computed_kpis_entity_period_start",
            "entity_name",
            "period_start",
        ),
    )
