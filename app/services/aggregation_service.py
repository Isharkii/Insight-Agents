"""
app/services/aggregation_service.py

Data aggregation layer for KPI calculations.

Translates raw CanonicalInsightRecord rows into clean numerical values
that KPIService calculation methods can consume directly.

Metric name conventions
-----------------------
Records are expected to be stored in CanonicalInsightRecord with the
following ``metric_name`` values (all under ``category="sales"``):

    recurring_revenue       – recurring revenue per subscription snapshot
    active_customer_count   – count of active customers at a point in time
    churned_customer_count  – count of customers lost in a reporting window

These constants are defined at module level and can be overridden when
constructing AggregationService if a deployment uses different names.

Query design
------------
Every public method issues exactly one SQL statement (or two clearly-bounded
scalar queries for ARPU, which aggregates over different time shapes).
There are no per-row follow-up queries — N+1 is structurally impossible.

No business logic lives here. Division, formula application, and result
structuring belong to KPIService.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Final, Sequence

from sqlalchemy import Numeric, Text, case, cast, func, select
from sqlalchemy.orm import Session

from db.models.canonical_insight_record import CanonicalInsightRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric name constants
# ---------------------------------------------------------------------------

METRIC_RECURRING_REVENUE: Final[str] = "recurring_revenue"
"""metric_name for per-subscription recurring revenue snapshots."""

METRIC_ACTIVE_CUSTOMER_COUNT: Final[str] = "active_customer_count"
"""metric_name for point-in-time active customer headcount."""

METRIC_CHURNED_CUSTOMER_COUNT: Final[str] = "churned_customer_count"
"""metric_name for customers lost within a reporting window."""

DEFAULT_KPI_CATEGORY: Final[str] = "sales"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numeric(col: object) -> object:
    """
    Cast a JSONB column that holds a plain scalar number to SQLAlchemy Numeric.

    PostgreSQL cannot cast JSONB directly to numeric; the intermediate
    text cast is required: ``metric_value::text::numeric``.
    """
    return cast(cast(col, Text()), Numeric())  # type: ignore[arg-type]


def _as_tuple(value: str | Sequence[str], *, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        normalized = value.strip()
        return (normalized,) if normalized else fallback
    normalized = tuple(str(item).strip() for item in value if str(item).strip())
    return normalized or fallback


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class AggregationService:
    """
    Reads CanonicalInsightRecord rows and produces clean numerical inputs
    for KPIService calculation methods.

    All methods are read-only and never mutate session state.

    Parameters
    ----------
    session:
        Active SQLAlchemy session. The caller controls its lifecycle.
    metric_recurring_revenue:
        Override the default ``metric_name`` for recurring revenue rows.
    metric_active_customer_count:
        Override the default ``metric_name`` for active-customer snapshots.
    metric_churned_customer_count:
        Override the default ``metric_name`` for churned-customer counts.
    """

    def __init__(
        self,
        session: Session,
        *,
        metric_recurring_revenue: str | Sequence[str] = METRIC_RECURRING_REVENUE,
        metric_active_customer_count: str | Sequence[str] = METRIC_ACTIVE_CUSTOMER_COUNT,
        metric_churned_customer_count: str | Sequence[str] = METRIC_CHURNED_CUSTOMER_COUNT,
        categories: str | Sequence[str] = DEFAULT_KPI_CATEGORY,
    ) -> None:
        self._session = session
        self._metric_revenue = _as_tuple(
            metric_recurring_revenue,
            fallback=(METRIC_RECURRING_REVENUE,),
        )
        self._metric_active = _as_tuple(
            metric_active_customer_count,
            fallback=(METRIC_ACTIVE_CUSTOMER_COUNT,),
        )
        self._metric_churned = _as_tuple(
            metric_churned_customer_count,
            fallback=(METRIC_CHURNED_CUSTOMER_COUNT,),
        )
        self._categories = _as_tuple(
            categories,
            fallback=(DEFAULT_KPI_CATEGORY,),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_period_revenue(
        self,
        entity_name: str,
        start_date: datetime,
        end_date: datetime,
    ) -> float | None:
        """
        Sum all recurring revenue records for *entity_name* within [start_date, end_date].

        Queries
        -------
        Single aggregation::

            SELECT SUM(metric_value::text::numeric)
            FROM   canonical_insight_records
            WHERE  entity_name    = :entity
              AND  category       = 'sales'
              AND  metric_name    = 'recurring_revenue'
              AND  timestamp BETWEEN :start AND :end

        Returns
        -------
        float | None
            Total revenue for the period. ``None`` when no matching records
            exist — callers must not treat missing data as zero.
        """
        numeric_value = _to_numeric(CanonicalInsightRecord.metric_value)

        # Count matching rows to distinguish "no data" from "sum is zero".
        count_stmt = select(func.count()).where(
            CanonicalInsightRecord.entity_name == entity_name,
            CanonicalInsightRecord.category.in_(self._categories),
            CanonicalInsightRecord.metric_name.in_(self._metric_revenue),
            CanonicalInsightRecord.timestamp >= start_date,
            CanonicalInsightRecord.timestamp < end_date,
        )
        row_count = self._session.scalar(count_stmt) or 0

        if row_count == 0:
            logger.debug(
                "get_period_revenue entity=%r [%s, %s] → None (no records)",
                entity_name, start_date.isoformat(), end_date.isoformat(),
            )
            return None

        stmt = select(func.coalesce(func.sum(numeric_value), 0)).where(
            CanonicalInsightRecord.entity_name == entity_name,
            CanonicalInsightRecord.category.in_(self._categories),
            CanonicalInsightRecord.metric_name.in_(self._metric_revenue),
            CanonicalInsightRecord.timestamp >= start_date,
            CanonicalInsightRecord.timestamp < end_date,
        )

        result = self._session.scalar(stmt)
        total = float(result) if result is not None else 0.0
        logger.debug(
            "get_period_revenue entity=%r [%s, %s] → %.4f",
            entity_name, start_date.isoformat(), end_date.isoformat(), total,
        )
        return total

    def get_active_customers(
        self,
        entity_name: str,
        start_date: datetime,
    ) -> int | None:
        """
        Return the most recent active-customer headcount at or before *start_date*.

        This is a point-in-time lookup: it fetches the single latest snapshot
        whose ``timestamp <= start_date``, which reflects the number of active
        customers at the beginning of a measurement period.

        Queries
        -------
        Single row fetch (LIMIT 1)::

            SELECT metric_value
            FROM   canonical_insight_records
            WHERE  entity_name = :entity
              AND  category    = 'sales'
              AND  metric_name = 'active_customer_count'
              AND  timestamp   <= :start_date
            ORDER BY timestamp DESC
            LIMIT 1

        Returns
        -------
        int | None
            Active customer count. ``None`` when no snapshot exists at or
            before *start_date* — callers must not treat missing data as zero.
        """
        stmt = (
            select(CanonicalInsightRecord.metric_value)
            .where(
                CanonicalInsightRecord.entity_name == entity_name,
                CanonicalInsightRecord.category.in_(self._categories),
                CanonicalInsightRecord.metric_name.in_(self._metric_active),
                CanonicalInsightRecord.timestamp <= start_date,
            )
            .order_by(CanonicalInsightRecord.timestamp.desc())
            .limit(1)
        )

        raw = self._session.scalar(stmt)
        if raw is None:
            logger.debug(
                "get_active_customers entity=%r at %s → None (no snapshot)",
                entity_name, start_date.isoformat(),
            )
            return None
        count = int(raw)
        logger.debug(
            "get_active_customers entity=%r at %s → %d",
            entity_name, start_date.isoformat(), count,
        )
        return count

    def get_lost_customers(
        self,
        entity_name: str,
        start_date: datetime,
        end_date: datetime,
    ) -> int | None:
        """
        Sum all churned-customer-count records within [start_date, end_date].

        Each CanonicalInsightRecord with ``metric_name="churned_customer_count"``
        represents the number of customers lost in a sub-window (e.g. one day
        or one week). Summing them yields the total for the requested period.

        Queries
        -------
        Single aggregation::

            SELECT SUM(metric_value::text::numeric)
            FROM   canonical_insight_records
            WHERE  entity_name = :entity
              AND  category    = 'sales'
              AND  metric_name = 'churned_customer_count'
              AND  timestamp BETWEEN :start AND :end

        Returns
        -------
        int | None
            Total customers lost in the period. ``None`` when no records
            exist — callers must not treat missing data as zero.
        """
        numeric_value = _to_numeric(CanonicalInsightRecord.metric_value)

        # Count matching rows to distinguish "no data" from "sum is zero".
        count_stmt = select(func.count()).where(
            CanonicalInsightRecord.entity_name == entity_name,
            CanonicalInsightRecord.category.in_(self._categories),
            CanonicalInsightRecord.metric_name.in_(self._metric_churned),
            CanonicalInsightRecord.timestamp >= start_date,
            CanonicalInsightRecord.timestamp < end_date,
        )
        row_count = self._session.scalar(count_stmt) or 0

        if row_count == 0:
            logger.debug(
                "get_lost_customers entity=%r [%s, %s] → None (no records)",
                entity_name, start_date.isoformat(), end_date.isoformat(),
            )
            return None

        stmt = select(func.coalesce(func.sum(numeric_value), 0)).where(
            CanonicalInsightRecord.entity_name == entity_name,
            CanonicalInsightRecord.category.in_(self._categories),
            CanonicalInsightRecord.metric_name.in_(self._metric_churned),
            CanonicalInsightRecord.timestamp >= start_date,
            CanonicalInsightRecord.timestamp < end_date,
        )

        result = self._session.scalar(stmt)
        lost = int(result) if result is not None else 0
        logger.debug(
            "get_lost_customers entity=%r [%s, %s] → %d",
            entity_name, start_date.isoformat(), end_date.isoformat(), lost,
        )
        return lost

    def get_average_revenue_per_user(
        self,
        entity_name: str,
        start_date: datetime,
        end_date: datetime,
    ) -> float | None:
        """
        Compute Average Revenue Per User (ARPU) for *entity_name* over the period.

        Formula::

            ARPU = total_recurring_revenue / active_customer_count_at_start

        ARPU aggregates over two different time shapes:

        * **Revenue** is summed over the full ``[start_date, end_date]`` window.
        * **Customer count** is the most recent headcount snapshot at or before
          ``start_date`` (point-in-time).

        Because these shapes differ, two targeted scalar queries are issued
        rather than a single mis-shaped aggregation. Both are single-pass and
        index-bound — not N+1.

        Returns
        -------
        float | None
            ARPU for the period. ``None`` when revenue data is missing or
            when no active-customer snapshot exists — callers must not treat
            missing data as zero.
        """
        total_revenue = self.get_period_revenue(entity_name, start_date, end_date)
        if total_revenue is None:
            logger.debug(
                "get_average_revenue_per_user entity=%r: revenue missing, ARPU=None",
                entity_name,
            )
            return None

        customer_count = self.get_active_customers(entity_name, start_date)
        if customer_count is None or customer_count == 0:
            logger.debug(
                "get_average_revenue_per_user entity=%r: active customers %s, ARPU=None",
                entity_name,
                "missing" if customer_count is None else "zero",
            )
            return None

        arpu = total_revenue / customer_count
        logger.debug(
            "get_average_revenue_per_user entity=%r [%s, %s] → %.4f "
            "(revenue=%.4f / customers=%d)",
            entity_name,
            start_date.isoformat(),
            end_date.isoformat(),
            arpu,
            total_revenue,
            customer_count,
        )
        return arpu

    # ------------------------------------------------------------------
    # Convenience: fetch all subscription revenue rows individually
    # ------------------------------------------------------------------

    def get_subscription_revenues(
        self,
        entity_name: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[float]:
        """
        Return individual recurring revenue values for each subscription row
        within [start_date, end_date].

        Use this when feeding :class:`~app.services.kpi_service.MRRInput`
        directly, as MRRInput expects a sequence of per-subscription amounts
        rather than a pre-summed total.

        Queries
        -------
        Single scan::

            SELECT metric_value
            FROM   canonical_insight_records
            WHERE  entity_name = :entity
              AND  category    = 'sales'
              AND  metric_name = 'recurring_revenue'
              AND  timestamp BETWEEN :start AND :end
            ORDER BY timestamp ASC

        Returns
        -------
        list[float]
            Per-subscription revenue values. Empty list when no rows exist.
        """
        stmt = (
            select(CanonicalInsightRecord.metric_value)
            .where(
                CanonicalInsightRecord.entity_name == entity_name,
                CanonicalInsightRecord.category.in_(self._categories),
                CanonicalInsightRecord.metric_name.in_(self._metric_revenue),
                CanonicalInsightRecord.timestamp >= start_date,
                CanonicalInsightRecord.timestamp < end_date,
            )
            .order_by(CanonicalInsightRecord.timestamp.asc())
        )

        rows = self._session.scalars(stmt).all()
        revenues = [float(r) for r in rows if r is not None]
        logger.debug(
            "get_subscription_revenues entity=%r [%s, %s] → %d rows",
            entity_name, start_date.isoformat(), end_date.isoformat(), len(revenues),
        )
        return revenues

    # ------------------------------------------------------------------
    # Pre-computed metric passthrough
    # ------------------------------------------------------------------

    def get_precomputed_metrics(
        self,
        entity_name: str,
        start_date: datetime,
        end_date: datetime,
        metric_names: Sequence[str],
    ) -> dict[str, float | None]:
        """
        Fetch pre-computed metric values directly from canonical records.

        Some datasets ship with already-derived metrics (e.g. ``churn_rate``,
        ``growth_rate``, ``ltv``) instead of the raw building blocks.  This
        method retrieves the latest value for each requested metric name within
        the time window, allowing the orchestrator to use them as fallbacks
        when the deterministic formula cannot compute a metric.

        Returns
        -------
        dict[str, float | None]
            Metric name → value.  ``None`` for metrics not found.
        """
        if not metric_names:
            return {}

        numeric_value = _to_numeric(CanonicalInsightRecord.metric_value)
        stmt = (
            select(
                CanonicalInsightRecord.metric_name,
                numeric_value,
            )
            .where(
                CanonicalInsightRecord.entity_name == entity_name,
                CanonicalInsightRecord.category.in_(self._categories),
                CanonicalInsightRecord.metric_name.in_(tuple(metric_names)),
                CanonicalInsightRecord.timestamp >= start_date,
                CanonicalInsightRecord.timestamp < end_date,
                CanonicalInsightRecord.metric_value.is_not(None),
            )
            .order_by(
                CanonicalInsightRecord.metric_name,
                CanonicalInsightRecord.timestamp.desc(),
            )
            .distinct(CanonicalInsightRecord.metric_name)
        )

        result: dict[str, float | None] = {name: None for name in metric_names}
        for row in self._session.execute(stmt).all():
            name, value = row[0], row[1]
            if value is not None:
                result[name] = float(value)

        logger.debug(
            "get_precomputed_metrics entity=%r [%s, %s] metrics=%s → found=%s",
            entity_name,
            start_date.isoformat(),
            end_date.isoformat(),
            list(metric_names),
            {k: v for k, v in result.items() if v is not None},
        )
        return result
