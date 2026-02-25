from __future__ import annotations

from typing import Any, Sequence

from app.services.statistics.normalization import coerce_numeric_series


def iqr_bounds(
    values: Sequence[Any],
    *,
    multiplier: float = 1.5,
) -> dict[str, float | None]:
    series = sorted(coerce_numeric_series(values))
    if len(series) < 2:
        return {
            "q1": None,
            "q3": None,
            "iqr": None,
            "lower_bound": None,
            "upper_bound": None,
        }

    q1 = _quantile(series, 0.25)
    q3 = _quantile(series, 0.75)
    iqr = q3 - q1
    m = max(0.5, float(multiplier))
    lower = q1 - (m * iqr)
    upper = q3 + (m * iqr)
    return {
        "q1": round(q1, 6),
        "q3": round(q3, 6),
        "iqr": round(iqr, 6),
        "lower_bound": round(lower, 6),
        "upper_bound": round(upper, 6),
    }


def detect_iqr_anomalies(
    values: Sequence[Any],
    *,
    multiplier: float = 1.5,
) -> dict[str, Any]:
    series = coerce_numeric_series(values)
    bounds = iqr_bounds(series, multiplier=multiplier)

    lower = bounds.get("lower_bound")
    upper = bounds.get("upper_bound")
    if lower is None or upper is None:
        return {
            "status": "insufficient_history",
            "anomaly_flags": [False for _ in series],
            "anomaly_indexes": [],
            "anomaly_values": [],
            "bounds": bounds,
        }

    low = float(lower)
    high = float(upper)
    flags: list[bool] = []
    indexes: list[int] = []
    values_out: list[float] = []
    for idx, value in enumerate(series):
        is_anomaly = value < low or value > high
        flags.append(is_anomaly)
        if is_anomaly:
            indexes.append(idx)
            values_out.append(round(value, 6))

    return {
        "status": "ok",
        "anomaly_flags": flags,
        "anomaly_indexes": indexes,
        "anomaly_values": values_out,
        "bounds": bounds,
    }


def _quantile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    p_clamped = max(0.0, min(1.0, float(p)))
    position = p_clamped * (len(sorted_values) - 1)
    lo = int(position)
    hi = min(lo + 1, len(sorted_values) - 1)
    weight = position - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)

