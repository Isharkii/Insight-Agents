"""
kpi/saas.py

SaaS KPI formula implementation.

Expected inputs
---------------
active_subscriptions : list[float]
    Recurring revenue amount for each active subscription in the period.
starting_customers : int
    Number of active customers at the beginning of the period.
lost_customers : int
    Number of customers who cancelled or lapsed during the period.
gross_margin : float
    Gross margin as a decimal fraction (e.g. 0.70 = 70 %).
previous_mrr : float
    MRR from the immediately preceding period.

Formulas
--------
MRR         = sum(active_subscriptions)
Churn Rate  = lost_customers / starting_customers
ARPU        = MRR / starting_customers
LTV         = (ARPU * gross_margin) / churn_rate
Growth Rate = (current_mrr - previous_mrr) / previous_mrr

Division-by-zero cases return None for the affected metric.
"""

from __future__ import annotations

from typing import Any

from kpi.base import BaseKPIFormula

_SENTINEL = None  # value stored when a metric cannot be computed


class SaaSKPIFormula(BaseKPIFormula):
    """
    Deterministic SaaS KPI calculations with safe division-by-zero handling.

    All arithmetic is self-contained.  No I/O, no logging, no side effects.
    """

    def calculate(self, inputs: dict[str, Any]) -> dict[str, float | None]:
        """
        Compute MRR, churn rate, LTV, and growth rate from *inputs*.

        Parameters
        ----------
        inputs:
            Dictionary containing the keys listed in the module docstring.

        Returns
        -------
        dict
            Keys: ``mrr``, ``churn_rate``, ``ltv``, ``growth_rate``.
            A metric is ``None`` when its formula produces a division by zero.
        """
        active_subscriptions: list[float] = inputs["active_subscriptions"]
        starting_customers: int = inputs["starting_customers"]
        lost_customers: int = inputs["lost_customers"]
        gross_margin: float = inputs["gross_margin"]
        previous_mrr: float = inputs["previous_mrr"]

        mrr = _mrr(active_subscriptions)
        churn_rate = _churn_rate(lost_customers, starting_customers)
        arpu = _arpu(mrr, starting_customers)
        ltv = _ltv(arpu, gross_margin, churn_rate)
        growth_rate = _growth_rate(mrr, previous_mrr)

        return {
            "mrr": mrr,
            "churn_rate": churn_rate,
            "ltv": ltv,
            "growth_rate": growth_rate,
        }


# ---------------------------------------------------------------------------
# Pure formula functions
# ---------------------------------------------------------------------------


def _mrr(active_subscriptions: list[float]) -> float:
    """MRR = sum of all active subscription revenues."""
    return sum(active_subscriptions)


def _churn_rate(lost_customers: int, starting_customers: int) -> float | None:
    """
    Churn Rate = lost_customers / starting_customers.

    Returns None when starting_customers is zero.
    """
    if starting_customers == 0:
        return _SENTINEL
    return lost_customers / starting_customers


def _arpu(mrr: float, starting_customers: int) -> float | None:
    """
    ARPU = MRR / starting_customers.

    Returns None when starting_customers is zero.
    """
    if starting_customers == 0:
        return _SENTINEL
    return mrr / starting_customers


def _ltv(
    arpu: float | None,
    gross_margin: float,
    churn_rate: float | None,
) -> float | None:
    """
    LTV = (ARPU * gross_margin) / churn_rate.

    Returns None when ARPU or churn_rate is None (upstream division by zero)
    or when churn_rate is zero.
    """
    if arpu is None or churn_rate is None or churn_rate == 0.0:
        return _SENTINEL
    return (arpu * gross_margin) / churn_rate


def _growth_rate(current_mrr: float, previous_mrr: float) -> float | None:
    """
    Growth Rate = (current_mrr - previous_mrr) / previous_mrr.

    Returns None when previous_mrr is zero.
    """
    if previous_mrr == 0.0:
        return _SENTINEL
    return (current_mrr - previous_mrr) / previous_mrr
