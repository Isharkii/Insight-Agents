"""
kpi/ecommerce.py

Ecommerce KPI formula implementation.

Expected inputs
---------------
orders : list[float]
    Revenue amount for each individual order in the period.
total_visitors : int
    Total number of site visitors in the period.
marketing_spend : float
    Total marketing expenditure in the period.
new_customers : int
    Number of first-time customers acquired in the period.
unique_customers : int
    Number of distinct customers who placed at least one order.
previous_revenue : float
    Total revenue from the immediately preceding period.

Formulas
--------
Revenue          = sum(orders)
AOV              = revenue / number_of_orders
Conversion Rate  = number_of_orders / total_visitors
CAC              = marketing_spend / new_customers
Purchase Freq    = number_of_orders / unique_customers
LTV              = AOV * purchase_frequency
Growth Rate      = (current_revenue - previous_revenue) / previous_revenue

Division-by-zero cases return None for the affected metric.
"""

from __future__ import annotations

from typing import Any

from kpi.base import BaseKPIFormula

_SENTINEL = None  # value stored when a metric cannot be computed


class EcommerceKPIFormula(BaseKPIFormula):
    """
    Deterministic ecommerce KPI calculations with safe division-by-zero handling.

    All arithmetic is self-contained.  No I/O, no logging, no side effects.
    """

    def calculate(self, inputs: dict[str, Any]) -> dict[str, float | None]:
        """
        Compute Revenue, AOV, Conversion Rate, CAC, Purchase Frequency,
        LTV, and Growth Rate from *inputs*.

        Parameters
        ----------
        inputs:
            Dictionary containing the keys listed in the module docstring.

        Returns
        -------
        dict
            Keys: ``revenue``, ``aov``, ``conversion_rate``, ``cac``,
            ``purchase_frequency``, ``ltv``, ``growth_rate``.
            A metric is ``None`` when its formula produces a division by zero.
        """
        orders: list[float] = inputs["orders"]
        total_visitors: int = inputs["total_visitors"]
        marketing_spend: float = inputs["marketing_spend"]
        new_customers: int = inputs["new_customers"]
        unique_customers: int = inputs["unique_customers"]
        previous_revenue: float = inputs["previous_revenue"]

        revenue = _revenue(orders)
        number_of_orders = len(orders)

        aov = _aov(revenue, number_of_orders)
        conversion_rate = _conversion_rate(number_of_orders, total_visitors)
        cac = _cac(marketing_spend, new_customers)
        purchase_frequency = _purchase_frequency(number_of_orders, unique_customers)
        ltv = _ltv(aov, purchase_frequency)
        growth_rate = _growth_rate(revenue, previous_revenue)

        return {
            "revenue": revenue,
            "aov": aov,
            "conversion_rate": conversion_rate,
            "cac": cac,
            "purchase_frequency": purchase_frequency,
            "ltv": ltv,
            "growth_rate": growth_rate,
        }


# ---------------------------------------------------------------------------
# Pure formula functions
# ---------------------------------------------------------------------------


def _revenue(orders: list[float]) -> float:
    """Revenue = sum of all order amounts."""
    return sum(orders)


def _aov(revenue: float, number_of_orders: int) -> float | None:
    """
    AOV = revenue / number_of_orders.

    Returns None when number_of_orders is zero.
    """
    if number_of_orders == 0:
        return _SENTINEL
    return revenue / number_of_orders


def _conversion_rate(number_of_orders: int, total_visitors: int) -> float | None:
    """
    Conversion Rate = number_of_orders / total_visitors.

    Returns None when total_visitors is zero.
    """
    if total_visitors == 0:
        return _SENTINEL
    return number_of_orders / total_visitors


def _cac(marketing_spend: float, new_customers: int) -> float | None:
    """
    CAC = marketing_spend / new_customers.

    Returns None when new_customers is zero.
    """
    if new_customers == 0:
        return _SENTINEL
    return marketing_spend / new_customers


def _purchase_frequency(number_of_orders: int, unique_customers: int) -> float | None:
    """
    Purchase Frequency = number_of_orders / unique_customers.

    Returns None when unique_customers is zero.
    """
    if unique_customers == 0:
        return _SENTINEL
    return number_of_orders / unique_customers


def _ltv(aov: float | None, purchase_frequency: float | None) -> float | None:
    """
    LTV = AOV * purchase_frequency.

    Returns None when either input is None (upstream division by zero).
    """
    if aov is None or purchase_frequency is None:
        return _SENTINEL
    return aov * purchase_frequency


def _growth_rate(current_revenue: float, previous_revenue: float) -> float | None:
    """
    Growth Rate = (current_revenue - previous_revenue) / previous_revenue.

    Returns None when previous_revenue is zero.
    """
    if previous_revenue == 0.0:
        return _SENTINEL
    return (current_revenue - previous_revenue) / previous_revenue
