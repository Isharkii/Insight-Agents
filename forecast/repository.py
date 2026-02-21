"""
forecast/repository.py

SQLAlchemy ORM model and repository for persisted forecast results.
No business or forecasting logic lives here.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    DateTime,
    Index,
    String,
    UniqueConstraint,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, Session, mapped_column

from db.base import Base


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class ForecastMetric(Base):
    """
    Persisted snapshot of a single forecast run for one entity / metric pair.

    Columns
    -------
    id            – surrogate primary key (UUID v4, server-side default).
    entity_name   – logical owner of the metric (client, product, region …).
    metric_name   – KPI being forecast (e.g. ``"monthly_revenue"``).
    period_end    – last date covered by the historical series that was used.
    forecast_data – full forecast payload as returned by a forecast model.
    created_at    – UTC timestamp set at insert time.

    Indexes
    -------
    Individual B-tree indexes on ``entity_name``, ``metric_name``, and
    ``period_end`` support single-column filtering.

    A composite index on ``(entity_name, metric_name, period_end)`` supports
    the canonical lookup pattern and is used by
    :meth:`ForecastRepository.get_latest_forecast`.
    """

    __tablename__ = "forecast_metric"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    entity_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    metric_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    forecast_data: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        UniqueConstraint(
            "entity_name",
            "metric_name",
            "period_end",
            name="uq_forecast_metric_entity_metric_period",
        ),
        Index(
            "ix_forecast_metric_entity_metric_period",
            "entity_name",
            "metric_name",
            "period_end",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<ForecastMetric id={self.id} "
            f"entity={self.entity_name!r} "
            f"metric={self.metric_name!r} "
            f"period_end={self.period_end.date()}>"
        )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class ForecastRepository:
    """
    Data-access layer for :class:`ForecastMetric`.

    Accepts a *session* at construction time so the caller controls the
    transaction boundary — the repository never commits or rolls back on its
    own.  This makes it safe to compose multiple repository calls inside a
    single ``with session.begin():`` block.

    Parameters
    ----------
    session:
        An active :class:`sqlalchemy.orm.Session`.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_forecast(
        self,
        entity_name: str,
        metric_name: str,
        period_end: datetime,
        forecast_data: dict,
    ) -> ForecastMetric:
        """
        Persist a new forecast snapshot and return the mapped instance.

        The row is added to the current session but **not** committed;
        the caller must commit the enclosing transaction.

        Parameters
        ----------
        entity_name:
            Logical owner of the metric.
        metric_name:
            KPI identifier.
        period_end:
            Last date of the historical series used for this forecast.
            Timezone-aware datetimes are stored as-is; naive datetimes are
            assumed to be UTC.
        forecast_data:
            Serialisable dictionary produced by a forecast model.

        Returns
        -------
        ForecastMetric
            The newly created (but not yet committed) ORM instance.
        """
        if period_end.tzinfo is None:
            period_end = period_end.replace(tzinfo=timezone.utc)

        record = ForecastMetric(
            entity_name=entity_name,
            metric_name=metric_name,
            period_end=period_end,
            forecast_data=forecast_data,
        )
        self._session.add(record)
        return record

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_latest_forecast(
        self,
        entity_name: str,
        metric_name: str,
    ) -> Optional[ForecastMetric]:
        """
        Return the most recently *created* forecast for a given
        entity / metric pair, or ``None`` if no rows exist.

        Ordering by ``created_at`` (not ``period_end``) ensures that a
        re-run for the same period returns the freshest result.

        Parameters
        ----------
        entity_name:
            Logical owner of the metric.
        metric_name:
            KPI identifier.

        Returns
        -------
        ForecastMetric or None
        """
        stmt = (
            select(ForecastMetric)
            .where(
                ForecastMetric.entity_name == entity_name,
                ForecastMetric.metric_name == metric_name,
            )
            .order_by(ForecastMetric.created_at.desc())
            .limit(1)
        )
        return self._session.scalars(stmt).first()
