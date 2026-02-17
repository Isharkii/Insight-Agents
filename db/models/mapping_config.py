"""
db/models/mapping_config.py

Persistent manual schema mappings for client CSV ingestion.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import Boolean, Index, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base, TimestampMixin


class MappingConfig(Base, TimestampMixin):
    __tablename__ = "mapping_configs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
        comment="Human-readable config name",
    )
    client_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Optional client identifier for scoped matching",
    )
    field_mapping_json: Mapped[dict[str, str]] = mapped_column(
        JSONB,
        nullable=False,
        comment="Canonical field -> source column overrides",
    )
    alias_overrides_json: Mapped[dict[str, list[str]] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Optional canonical field alias overrides",
    )
    notes: Mapped[str | None] = mapped_column(String(500), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        UniqueConstraint("name", "client_name", name="uq_mapping_configs_name_client_name"),
        Index("ix_mapping_configs_name", "name"),
        Index("ix_mapping_configs_client_name", "client_name"),
        Index("ix_mapping_configs_is_active", "is_active"),
        Index("ix_mapping_configs_client_active", "client_name", "is_active"),
    )
