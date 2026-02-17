"""
db/models/client.py

Client model — root entity representing one business or organization.
All datasets and insights are scoped to a client.
"""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Index, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from db.models.dataset import Dataset


class Client(Base, TimestampMixin):
    """
    Represents a business or organization using the Insight Agent.

    One client owns many datasets, which in turn own metrics and insights.
    config stores optional per-client overrides (e.g., preferred KPIs, domain rules).
    """

    __tablename__ = "clients"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    domain: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Industry or vertical (e.g., retail, fintech, saas)",
    )

    config: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Per-client configuration overrides",
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Soft-disable a client without deletion",
    )

    # ── Relationships ──────────────────────────────────────────────────────────

    datasets: Mapped[list["Dataset"]] = relationship(
        "Dataset",
        back_populates="client",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    # ── Indexes ────────────────────────────────────────────────────────────────

    __table_args__ = (
        Index("ix_clients_domain", "domain"),
        Index("ix_clients_is_active", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<Client id={self.id} name={self.name!r} domain={self.domain!r}>"
