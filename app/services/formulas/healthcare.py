from __future__ import annotations

import math
from typing import Any

from kpi.base import BaseKPIFormula


class HealthcareFormula(BaseKPIFormula):
    """
    Deterministic healthcare KPI formula set.
    """

    def calculate(self, inputs: dict[str, Any]) -> dict[str, float | None]:
        service_revenue = _as_float_list(inputs.get("service_revenue", []))
        active_patients = _as_int(inputs.get("active_patients"))
        readmissions = _as_int(inputs.get("readmissions"))
        previous_revenue = _as_float(inputs.get("previous_revenue"))
        occupied_beds = _as_optional_float(inputs.get("occupied_beds"))
        total_beds = _as_optional_float(inputs.get("total_beds"))
        staff_hours = _as_optional_float(inputs.get("staff_hours"))

        patient_revenue = float(sum(service_revenue))
        readmission_rate = _safe_divide(float(readmissions), float(active_patients))
        growth_rate = _safe_growth(patient_revenue, previous_revenue)

        bed_occupancy_rate = _safe_divide(occupied_beds, total_beds)
        revenue_per_staff_hour = _safe_divide(patient_revenue, staff_hours)

        return {
            "patient_revenue": patient_revenue,
            "readmission_rate": readmission_rate,
            "growth_rate": growth_rate,
            "bed_occupancy_rate": bed_occupancy_rate,
            "revenue_per_staff_hour": revenue_per_staff_hour,
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
