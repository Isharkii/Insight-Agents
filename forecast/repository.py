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
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, Session, mapped_column

from db.base import Base
from db.repositories.entity_scope import normalize_tenant_id, resolve_entity_scope


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
    Individual B-tree indexes on ``tenant_id``, ``entity_id``, ``entity_name``,
    ``metric_name``, and ``period_end`` support filtered access paths.

    A composite index on ``(tenant_id, entity_id, metric_name, period_end)``
    supports tenant-isolated lookup and is used by
    :meth:`ForecastRepository.get_latest_forecast`.
    """

    __tablename__ = "forecast_metric"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    tenant_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        server_default="legacy",
    )
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenant_entities.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
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
            "tenant_id",
            "entity_id",
            "metric_name",
            "period_end",
            name="uq_forecast_metric_tenant_entity_metric_period",
        ),
        Index(
            "ix_forecast_metric_tenant_entity_metric_period",
            "tenant_id",
            "entity_id",
            "metric_name",
            "period_end",
        ),
        Index(
            "ix_forecast_metric_tenant_entity_name_metric_period",
            "tenant_id",
            "entity_name",
            "metric_name",
            "period_end",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<ForecastMetric id={self.id} "
            f"tenant={self.tenant_id!r} "
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
        *,
        tenant_id: str = "legacy",
        entity_id: uuid.UUID | None = None,
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
        tenant_id:
            Owning tenant identifier for isolation.
        entity_id:
            Optional stable entity identifier. When omitted, resolved from
            ``tenant_id + entity_name``.

        Returns
        -------
        ForecastMetric
            The newly created (but not yet committed) ORM instance.
        """
        if period_end.tzinfo is None:
            period_end = period_end.replace(tzinfo=timezone.utc)

        scope = resolve_entity_scope(
            self._session,
            tenant_id=tenant_id,
            entity_name=entity_name,
            entity_id=entity_id,
            create_if_missing=True,
        )

        record = ForecastMetric(
            tenant_id=scope.tenant_id,
            entity_id=scope.entity_id,
            entity_name=scope.entity_name,
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
        *,
        tenant_id: str = "legacy",
        entity_id: uuid.UUID | None = None,
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
        tenant_id:
            Owning tenant identifier for isolation.
        entity_id:
            Optional stable entity identifier for exact entity scope.

        Returns
        -------
        ForecastMetric or None
        """
        stmt = select(ForecastMetric).where(
            ForecastMetric.tenant_id == normalize_tenant_id(tenant_id),
            ForecastMetric.metric_name == metric_name,
        )
        if entity_id is not None:
            stmt = stmt.where(ForecastMetric.entity_id == entity_id)
        else:
            stmt = stmt.where(ForecastMetric.entity_name == entity_name)
        stmt = stmt.order_by(ForecastMetric.created_at.desc()).limit(1)
        return self._session.scalars(stmt).first()
