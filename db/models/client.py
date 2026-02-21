"""
db/models/client.py

Client model representing one business account.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Index, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from db.models.analytical_metric import AnalyticalMetric
    from db.models.dataset import Dataset
    from db.models.insight import Insight


class Client(Base, TimestampMixin):
    __tablename__ = "clients"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    domain: Mapped[str | None] = mapped_column(String(120), nullable=True)
    config: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Optional client-level configuration",
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    datasets: Mapped[list["Dataset"]] = relationship(
        "Dataset",
        back_populates="client",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    analytical_metrics: Mapped[list["AnalyticalMetric"]] = relationship(
        "AnalyticalMetric",
        back_populates="client",
        passive_deletes=True,
    )
    insights: Mapped[list["Insight"]] = relationship(
        "Insight",
        back_populates="client",
        passive_deletes=True,
    )

    __table_args__ = (
        UniqueConstraint("name", name="uq_clients_name"),
        Index("ix_clients_name", "name"),
        Index("ix_clients_domain", "domain"),
        Index("ix_clients_is_active", "is_active"),
    )
