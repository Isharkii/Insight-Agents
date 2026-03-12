"""
db/models/tenant_entity.py

Tenant-scoped entity registry used by analytical persistence tables.

This table decouples stable internal entity identity (entity_id) from the
human-readable entity label used in upstream payloads.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Index, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class TenantEntity(Base):
    """Canonical tenant-local entity identity."""

    __tablename__ = "tenant_entities"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    tenant_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Owning tenant identifier from auth/security context.",
    )
    entity_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Stable tenant-local entity key (legacy source: entity_name).",
    )
    display_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Human-readable label for UI/reporting.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "entity_key",
            name="uq_tenant_entities_tenant_entity_key",
        ),
        Index("ix_tenant_entities_tenant_id", "tenant_id"),
        Index("ix_tenant_entities_tenant_display_name", "tenant_id", "display_name"),
    )

