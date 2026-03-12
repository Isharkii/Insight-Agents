from __future__ import annotations

import math
from statistics import median
from typing import Any, Sequence

from app.services.statistics.normalization import coerce_numeric_series
from app.services.statistics.seasonality import detect_seasonality

_ZERO_GUARD = 1e-9


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
            "method": "iqr",
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
        "method": "iqr",
        "anomaly_flags": flags,
        "anomaly_indexes": indexes,
        "anomaly_values": values_out,
        "bounds": bounds,
    }


def mad_bounds(
    values: Sequence[Any],
    *,
    threshold: float = 3.5,
) -> dict[str, float | None]:
    series = coerce_numeric_series(values)
    if len(series) < 3:
        return {
            "median": None,
            "mad": None,
            "threshold": round(float(threshold), 6),
        }
    med = float(median(series))
    abs_dev = [abs(v - med) for v in series]
    mad = float(median(abs_dev))
    return {
        "median": round(med, 6),
        "mad": round(mad, 6),
        "threshold": round(float(threshold), 6),
    }


def detect_mad_anomalies(
    values: Sequence[Any],
    *,
    threshold: float = 3.5,
) -> dict[str, Any]:
    series = coerce_numeric_series(values)
    stats = mad_bounds(series, threshold=threshold)
    med = stats.get("median")
    mad = stats.get("mad")
    if med is None or mad is None:
        return {
            "status": "insufficient_history",
            "method": "mad",
            "anomaly_flags": [False for _ in series],
            "anomaly_indexes": [],
            "anomaly_values": [],
            "bounds": stats,
        }
    med_f = float(med)
    mad_f = max(float(mad), _ZERO_GUARD)
    flags: list[bool] = []
    indexes: list[int] = []
    values_out: list[float] = []

    for idx, value in enumerate(series):
        modified_z = 0.6745 * (value - med_f) / mad_f
        is_anomaly = abs(modified_z) >= threshold
        flags.append(is_anomaly)
        if is_anomaly:
            indexes.append(idx)
            values_out.append(round(value, 6))

    return {
        "status": "ok",
        "method": "mad",
        "anomaly_flags": flags,
        "anomaly_indexes": indexes,
        "anomaly_values": values_out,
        "bounds": stats,
    }


def detect_cusum_shifts(
    values: Sequence[Any],
    *,
    threshold_sigma: float = 4.0,
) -> dict[str, Any]:
    series = coerce_numeric_series(values)
    n = len(series)
    if n < 8:
        return {
            "status": "insufficient_history",
            "method": "cusum",
            "shift_detected": False,
            "changepoint_indexes": [],
            "cusum_trace": [],
        }

    mu = sum(series) / float(n)
    variance = sum((v - mu) ** 2 for v in series) / float(n)
    sigma = math.sqrt(max(variance, _ZERO_GUARD))
    threshold = max(0.1, float(threshold_sigma)) * sigma

    trace: list[float] = [0.0]
    changepoints: list[int] = []
    s = 0.0
    for idx in range(1, n):
        s += (series[idx] - mu)
        trace.append(round(s, 6))
        if abs(s) >= threshold:
            changepoints.append(idx)
            s = 0.0

    return {
        "status": "ok",
        "method": "cusum",
        "shift_detected": bool(changepoints),
        "changepoint_indexes": changepoints,
        "cusum_trace": trace,
        "threshold": round(threshold, 6),
    }


def detect_anomalies(
    values: Sequence[Any],
    *,
    iqr_multiplier: float = 1.5,
    mad_threshold: float = 3.5,
    cusum_threshold_sigma: float = 4.0,
) -> dict[str, Any]:
    """
    Multi-method anomaly detection with simple type classification.
    """
    series = coerce_numeric_series(values)
    iqr = detect_iqr_anomalies(series, multiplier=iqr_multiplier)
    mad = detect_mad_anomalies(series, threshold=mad_threshold)
    cusum = detect_cusum_shifts(series, threshold_sigma=cusum_threshold_sigma)
    seasonal = _detect_seasonal_residual_anomalies(series)

    iqr_set = set(iqr.get("anomaly_indexes", []))
    mad_set = set(mad.get("anomaly_indexes", []))
    cusum_set = set(cusum.get("changepoint_indexes", []))
    seasonal_set = set(seasonal.get("anomaly_indexes", []))
    all_indexes = sorted(iqr_set | mad_set | cusum_set | seasonal_set)

    classified: list[dict[str, Any]] = []
    for idx in all_indexes:
        if idx in cusum_set:
            anomaly_type = "level_shift"
        elif idx in seasonal_set:
            anomaly_type = "seasonal_anomaly"
        else:
            anomaly_type = "transient_spike"
        value = series[idx] if 0 <= idx < len(series) else None
        classified.append(
            {
                "index": idx,
                "value": round(float(value), 6) if isinstance(value, (int, float)) else None,
                "type": anomaly_type,
            }
        )

    flags = [False for _ in series]
    for idx in all_indexes:
        if 0 <= idx < len(flags):
            flags[idx] = True
    values_out = [
        round(series[idx], 6)
        for idx in all_indexes
        if 0 <= idx < len(series)
    ]

    by_type = {"level_shift": 0, "transient_spike": 0, "seasonal_anomaly": 0}
    for item in classified:
        anomaly_type = str(item.get("type") or "")
        if anomaly_type in by_type:
            by_type[anomaly_type] += 1

    status = (
        "insufficient_history"
        if len(series) < 3
        else "ok"
    )
    return {
        "status": status,
        "method": "ensemble",
        "anomaly_flags": flags,
        "anomaly_indexes": all_indexes,
        "anomaly_values": values_out,
        "classified_anomalies": classified,
        "summary": {
            "total_anomalies": len(all_indexes),
            "by_type": by_type,
        },
        "methods": {
            "iqr": iqr,
            "mad": mad,
            "cusum": cusum,
            "seasonal": seasonal,
        },
    }


def _detect_seasonal_residual_anomalies(values: Sequence[float]) -> dict[str, Any]:
    series = [float(v) for v in values if _is_finite(v)]
    n = len(series)
    if n < 8:
        return {
            "status": "insufficient_history",
            "method": "seasonal_residual",
            "anomaly_indexes": [],
            "anomaly_values": [],
        }
    seasonality = detect_seasonality(series)
    period = seasonality.get("primary_period")
    if not seasonality.get("detected") or not isinstance(period, int) or period < 2:
        return {
            "status": "none",
            "method": "seasonal_residual",
            "anomaly_indexes": [],
            "anomaly_values": [],
        }

    phases: list[list[float]] = [[] for _ in range(period)]
    for idx, value in enumerate(series):
        phases[idx % period].append(value)
    phase_means = [
        (sum(bucket) / float(len(bucket))) if bucket else 0.0
        for bucket in phases
    ]
    residuals = [
        series[idx] - phase_means[idx % period]
        for idx in range(n)
    ]
    mu = sum(residuals) / float(n)
    variance = sum((r - mu) ** 2 for r in residuals) / float(n)
    sigma = math.sqrt(max(variance, _ZERO_GUARD))
    if sigma < _ZERO_GUARD:
        return {
            "status": "none",
            "method": "seasonal_residual",
            "anomaly_indexes": [],
            "anomaly_values": [],
        }

    indexes: list[int] = []
    values_out: list[float] = []
    for idx, residual in enumerate(residuals):
        z = abs((residual - mu) / sigma)
        if z >= 2.5:
            indexes.append(idx)
            values_out.append(round(series[idx], 6))
    return {
        "status": "ok",
        "method": "seasonal_residual",
        "anomaly_indexes": indexes,
        "anomaly_values": values_out,
        "period": period,
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


def _is_finite(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)

