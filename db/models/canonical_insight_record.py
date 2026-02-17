"""
db/models/canonical_insight_record.py

Canonical normalized record for cross-source insight ingestion.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class CanonicalSourceType:
    CSV = "csv"
    API = "api"
    SCRAPE = "scrape"


class CanonicalCategory:
    SALES = "sales"
    MARKETING = "marketing"
    PRICING = "pricing"
    EVENT = "event"
    MACRO = "macro"


class CanonicalInsightRecord(Base):
    __tablename__ = "canonical_insight_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    source_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="csv, api, scrape",
    )
    entity_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Client or competitor entity name",
    )
    category: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="sales, marketing, pricing, event, macro",
    )
    metric_name: Mapped[str] = mapped_column(String(120), nullable=False)
    metric_value: Mapped[Any] = mapped_column(
        JSONB,
        nullable=False,
        comment="Canonical metric value (number, string, object, or array)",
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    region: Mapped[str | None] = mapped_column(String(120), nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Additional source-specific metadata",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        Index("ix_canonical_insight_records_source_type", "source_type"),
        Index("ix_canonical_insight_records_entity_name", "entity_name"),
        Index("ix_canonical_insight_records_category", "category"),
        Index("ix_canonical_insight_records_metric_name", "metric_name"),
        Index("ix_canonical_insight_records_timestamp", "timestamp"),
        Index("ix_canonical_insight_records_region", "region"),
        Index(
            "ix_canonical_insight_records_entity_category_timestamp",
            "entity_name",
            "category",
            "timestamp",
        ),
    )
