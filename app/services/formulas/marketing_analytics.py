from __future__ import annotations

import math
from typing import Any

from kpi.base import BaseKPIFormula


class MarketingAnalyticsFormula(BaseKPIFormula):
    """
    Deterministic marketing analytics KPI formula set.
    """

    def calculate(self, inputs: dict[str, Any]) -> dict[str, float | None]:
        campaign_revenue = _as_float_list(inputs.get("campaign_revenue", []))
        conversions = _as_int(inputs.get("conversions"))
        churned_customers = _as_int(inputs.get("churned_customers"))
        previous_revenue = _as_float(inputs.get("previous_revenue"))
        ad_spend = _as_optional_float(inputs.get("ad_spend"))
        impressions = _as_optional_float(inputs.get("impressions"))
        clicks = _as_optional_float(inputs.get("clicks"))

        attributed_revenue = float(sum(campaign_revenue))
        pipeline_churn = _safe_divide(float(churned_customers), float(conversions))
        growth_rate = _safe_growth(attributed_revenue, previous_revenue)

        roas = _safe_divide(attributed_revenue, ad_spend)
        ctr = _safe_divide(clicks, impressions)
        cac = _safe_divide(ad_spend, float(conversions))

        return {
            "attributed_revenue": attributed_revenue,
            "pipeline_churn": pipeline_churn,
            "growth_rate": growth_rate,
            "roas": roas,
            "ctr": ctr,
            "cac": cac,
        }


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _safe_growth(current: float, previous: float) -> float | None:
    if previous == 0.0:
        return None
    return (current - previous) / previous


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
