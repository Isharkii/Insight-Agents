"""
app/services/kpi_service.py

Deterministic KPI calculation engine.

All calculation functions operate on pre-fetched, typed input structures.
No database logic lives inside the calculation layer — the caller is
responsible for querying CanonicalInsightRecord rows and mapping them
into the appropriate input dataclasses before invoking this service.

Formulas
--------
MRR           = sum of recurring_revenue for all active subscriptions
Churn Rate    = customers_lost / customers_at_start
LTV           = average_revenue_per_user / churn_rate
Growth Rate   = (current_period_revenue - previous_period_revenue)
                / previous_period_revenue
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MRRInput:
    """
    Pre-fetched recurring revenue figures for active subscriptions.

    The caller queries CanonicalInsightRecord with
    ``category="sales"`` and ``metric_name="recurring_revenue"``
    for the desired period, filters to active subscriptions, and
    populates ``active_subscription_revenues``.
    """

    active_subscription_revenues: Sequence[float]
    """Recurring revenue amount for each active subscription in the period."""


@dataclass(frozen=True)
class ChurnInput:
    """
    Customer headcount data required for churn rate calculation.

    The caller derives these counts from CanonicalInsightRecord rows
    that track subscription status transitions within the period.
    """

    customers_at_start: int
    """Number of active customers at the beginning of the period."""

    customers_lost: int
    """Number of customers who cancelled or lapsed during the period."""


@dataclass(frozen=True)
class LTVInput:
    """
    Aggregated revenue and churn data for LTV calculation.

    Typical construction:
      - ``average_revenue_per_user`` = MRR / total active customers
      - ``churn_rate`` from :class:`ChurnInput` via :meth:`KPIService.calculate_churn`
    """

    average_revenue_per_user: float
    """Mean revenue generated per active user in the period."""

    churn_rate: float
    """Churn rate as a decimal fraction (e.g. 0.05 = 5 %)."""


@dataclass(frozen=True)
class GrowthRateInput:
    """
    Revenue totals for two consecutive periods.

    The caller sums ``metric_value`` from CanonicalInsightRecord rows
    for each period window and populates both fields.
    """

    current_period_revenue: float
    """Total revenue recognised in the current measurement period."""

    previous_period_revenue: float
    """Total revenue recognised in the immediately preceding period."""


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KPIResult:
    """
    Structured result returned by every KPI calculation method.

    ``value`` is ``None`` when the calculation cannot be completed
    (e.g. division by zero). Inspect ``error`` for the reason.
    """

    metric: str
    """Human-readable name of the KPI (e.g. ``"mrr"``, ``"churn_rate"``)."""

    value: float | None
    """Computed KPI value, or ``None`` if the calculation failed."""

    unit: str
    """Unit of measurement (e.g. ``"currency"``, ``"rate"``, ``"ratio"``)."""

    computed_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    """UTC timestamp of when the result was produced."""

    error: str | None = None
    """Populated with a short description when ``value`` is ``None``."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class KPIService:
    """
    Stateless, deterministic KPI calculation engine.

    All methods accept typed input dataclasses whose values have been
    pre-fetched from the database by the caller.  No I/O is performed
    inside this class — it contains only pure arithmetic business logic.

    Usage::

        service = KPIService()
        result = service.calculate_mrr(MRRInput(active_subscription_revenues=[500.0, 300.0]))
        print(result.value)  # 800.0
    """

    # ------------------------------------------------------------------
    # MRR
    # ------------------------------------------------------------------

    def calculate_mrr(self, data: MRRInput) -> KPIResult:
        """
        Calculate Monthly Recurring Revenue (MRR).

        Formula::

            MRR = Σ recurring_revenue  for all active subscriptions in the period

        Parameters
        ----------
        data:
            Pre-fetched recurring revenue values for every active
            subscription in the target period.

        Returns
        -------
        KPIResult
            ``metric="mrr"``, ``unit="currency"``.
            Value is ``0.0`` for an empty subscription list (valid state).
        """
        revenues = list(data.active_subscription_revenues)
        mrr = sum(revenues)
        logger.debug("MRR computed from %d subscriptions: %.4f", len(revenues), mrr)
        return KPIResult(metric="mrr", value=mrr, unit="currency")

    # ------------------------------------------------------------------
    # Churn Rate
    # ------------------------------------------------------------------

    def calculate_churn(self, data: ChurnInput) -> KPIResult:
        """
        Calculate the customer churn rate for a period.

        Formula::

            Churn Rate = customers_lost / customers_at_start

        Edge cases
        ----------
        * ``customers_at_start == 0`` → cannot divide; returns ``None`` with
          an ``error`` message.

        Parameters
        ----------
        data:
            Headcount figures derived from CanonicalInsightRecord subscription
            status transitions for the measurement period.

        Returns
        -------
        KPIResult
            ``metric="churn_rate"``, ``unit="rate"`` (decimal fraction 0–1).
        """
        if data.customers_at_start == 0:
            logger.warning("Churn rate calculation skipped: customers_at_start is zero.")
            return KPIResult(
                metric="churn_rate",
                value=None,
                unit="rate",
                error="Division by zero: customers_at_start must be > 0.",
            )

        rate = data.customers_lost / data.customers_at_start
        logger.debug(
            "Churn rate computed: %.6f (%d lost / %d at start)",
            rate,
            data.customers_lost,
            data.customers_at_start,
        )
        return KPIResult(metric="churn_rate", value=rate, unit="rate")

    # ------------------------------------------------------------------
    # LTV
    # ------------------------------------------------------------------

    def calculate_ltv(self, data: LTVInput) -> KPIResult:
        """
        Calculate Customer Lifetime Value (LTV).

        Formula::

            LTV = Average Revenue Per User (ARPU) / Churn Rate

        This is the standard simplified LTV model: it estimates the total
        revenue a business can expect from one customer before they churn.

        Edge cases
        ----------
        * ``churn_rate == 0.0`` → cannot divide; returns ``None`` with an
          ``error`` message.  A zero churn rate would imply infinite LTV,
          which is not a meaningful reportable value.

        Parameters
        ----------
        data:
            ARPU and churn rate (as a decimal fraction, e.g. ``0.05``).

        Returns
        -------
        KPIResult
            ``metric="ltv"``, ``unit="currency"``.
        """
        if data.churn_rate == 0.0:
            logger.warning("LTV calculation skipped: churn_rate is zero (implies infinite LTV).")
            return KPIResult(
                metric="ltv",
                value=None,
                unit="currency",
                error="Division by zero: churn_rate must be > 0 to compute a finite LTV.",
            )

        ltv = data.average_revenue_per_user / data.churn_rate
        logger.debug(
            "LTV computed: %.4f (ARPU=%.4f / churn=%.6f)",
            ltv,
            data.average_revenue_per_user,
            data.churn_rate,
        )
        return KPIResult(metric="ltv", value=ltv, unit="currency")

    # ------------------------------------------------------------------
    # Growth Rate
    # ------------------------------------------------------------------

    def calculate_growth_rate(self, data: GrowthRateInput) -> KPIResult:
        """
        Calculate period-over-period revenue growth rate.

        Formula::

            Growth Rate = (current_period_revenue - previous_period_revenue)
                          / previous_period_revenue

        A positive value indicates growth; a negative value indicates decline.

        Edge cases
        ----------
        * ``previous_period_revenue == 0.0`` → cannot divide; returns ``None``
          with an ``error`` message.  A zero baseline makes percentage growth
          undefined.

        Parameters
        ----------
        data:
            Total revenue for the current and the immediately preceding period.

        Returns
        -------
        KPIResult
            ``metric="growth_rate"``, ``unit="rate"`` (decimal fraction,
            e.g. ``0.25`` = 25 % growth).
        """
        if data.previous_period_revenue == 0.0:
            logger.warning(
                "Growth rate calculation skipped: previous_period_revenue is zero."
            )
            return KPIResult(
                metric="growth_rate",
                value=None,
                unit="rate",
                error="Division by zero: previous_period_revenue must be != 0.",
            )

        rate = (
            data.current_period_revenue - data.previous_period_revenue
        ) / data.previous_period_revenue
        logger.debug(
            "Growth rate computed: %.6f (current=%.4f, previous=%.4f)",
            rate,
            data.current_period_revenue,
            data.previous_period_revenue,
        )
        return KPIResult(metric="growth_rate", value=rate, unit="rate")
