from __future__ import annotations

import math
from typing import Any

from kpi.base import BaseKPIFormula


class OperationsFormula(BaseKPIFormula):
    """
    Deterministic operations KPI formula set.
    """

    def calculate(self, inputs: dict[str, Any]) -> dict[str, float | None]:
        throughput_series = _as_float_list(inputs.get("throughput_series", []))
        active_units = _as_int(inputs.get("active_units"))
        failed_units = _as_int(inputs.get("failed_units"))
        previous_output = _as_float(inputs.get("previous_output"))
        labor_hours = _as_optional_float(inputs.get("labor_hours"))
        downtime_hours = _as_optional_float(inputs.get("downtime_hours"))
        capacity_hours = _as_optional_float(inputs.get("capacity_hours"))

        throughput = float(sum(throughput_series))
        defect_rate = _safe_divide(float(failed_units), float(active_units))
        growth_rate = _safe_growth(throughput, previous_output)

        productivity = _safe_divide(throughput, labor_hours)
        uptime_rate = _safe_uptime(capacity_hours, downtime_hours)

        return {
            "throughput": throughput,
            "defect_rate": defect_rate,
            "growth_rate": growth_rate,
            "productivity": productivity,
            "uptime_rate": uptime_rate,
        }


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _safe_growth(current: float, previous: float) -> float | None:
    if previous == 0.0:
        return None
    return (current - previous) / previous


def _safe_uptime(capacity_hours: float | None, downtime_hours: float | None) -> float | None:
    if capacity_hours is None or downtime_hours is None or capacity_hours == 0.0:
        return None
    return (capacity_hours - downtime_hours) / capacity_hours


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
