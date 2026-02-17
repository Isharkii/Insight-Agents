"""
db/base.py

Declarative base and shared mixins for all SQLAlchemy models.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """
    Project-wide declarative base.
    All models must inherit from this class.
    """

    type_annotation_map: dict[type, Any] = {}


class TimestampMixin:
    """
    Mixin that adds created_at and updated_at to any model.
    updated_at is automatically refreshed on every UPDATE via onupdate.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),
    )
