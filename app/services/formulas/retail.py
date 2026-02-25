from __future__ import annotations

import math
from typing import Any

from kpi.base import BaseKPIFormula


class RetailFormula(BaseKPIFormula):
    """
    Deterministic retail KPI formula set.
    """

    def calculate(self, inputs: dict[str, Any]) -> dict[str, float | None]:
        sales_series = _as_float_list(inputs.get("sales_series", []))
        active_customers = _as_int(inputs.get("active_customers"))
        churned_customers = _as_int(inputs.get("churned_customers"))
        previous_sales = _as_float(inputs.get("previous_sales"))
        footfall = _as_optional_float(inputs.get("footfall"))
        orders = _as_optional_float(inputs.get("orders"))
        cogs = _as_optional_float(inputs.get("cogs"))

        net_sales = float(sum(sales_series))
        customer_churn = _safe_divide(float(churned_customers), float(active_customers))
        growth_rate = _safe_growth(net_sales, previous_sales)

        conversion_rate = _safe_divide(orders, footfall)
        gross_margin_rate = _safe_gross_margin_rate(net_sales, cogs)
        average_ticket = _safe_divide(net_sales, orders)

        return {
            "net_sales": net_sales,
            "customer_churn": customer_churn,
            "growth_rate": growth_rate,
            "conversion_rate": conversion_rate,
            "gross_margin_rate": gross_margin_rate,
            "average_ticket": average_ticket,
        }


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _safe_growth(current: float, previous: float) -> float | None:
    if previous == 0.0:
        return None
    return (current - previous) / previous


def _safe_gross_margin_rate(net_sales: float, cogs: float | None) -> float | None:
    if cogs is None or net_sales == 0.0:
        return None
    return (net_sales - cogs) / net_sales


def _as_float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        number = _as_optional_float(item)
        if number is not None and math.isfinite(number):
            out.append(float(number))
    return out


def _as_optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _as_float(value: Any) -> float:
    parsed = _as_optional_float(value)
    return parsed if parsed is not None else 0.0


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
