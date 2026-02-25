from __future__ import annotations

import math
from statistics import pstdev
from typing import Any

from kpi.base import BaseKPIFormula


class GeneralTimeseriesFormula(BaseKPIFormula):
    """
    Deterministic KPI set for generic time-series data.
    """

    def calculate(self, inputs: dict[str, Any]) -> dict[str, float | None]:
        values = _as_float_list(inputs.get("values", []))
        active_entities = _as_int(inputs.get("active_entities"))
        churned_entities = _as_int(inputs.get("churned_entities"))
        previous_value = _as_float(inputs.get("previous_value"))
        baseline_target = _as_optional_float(inputs.get("baseline_target"))

        timeseries_value = float(sum(values))
        churn_rate = _safe_divide(float(churned_entities), float(active_entities))
        growth_rate = _safe_growth(timeseries_value, previous_value)

        volatility = float(pstdev(values)) if len(values) >= 2 else None
        target_attainment = _safe_divide(timeseries_value, baseline_target)

        return {
            "timeseries_value": timeseries_value,
            "churn_rate": churn_rate,
            "growth_rate": growth_rate,
            "volatility": volatility,
            "target_attainment": target_attainment,
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
        parsed = _as_optional_float(item)
        if parsed is not None and math.isfinite(parsed):
            out.append(float(parsed))
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
