from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Sequence


_BUSINESS_RULES_PATH = Path(__file__).resolve().parents[3] / "config" / "business_rules.yaml"


@lru_cache(maxsize=1)
def _load_business_rules() -> dict[str, Any]:
    try:
        raw = _BUSINESS_RULES_PATH.read_text(encoding="utf-8")
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except (OSError, TypeError, ValueError):
        return {}


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: object, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


_STATISTICS_RULES = _as_dict(_load_business_rules().get("statistics"))
_STATISTICS_DEFAULTS = _as_dict(_STATISTICS_RULES.get("defaults"))
_STATISTICS_METRICS = _as_dict(_STATISTICS_RULES.get("metrics"))


@dataclass(frozen=True)
class MetricStatisticsConfig:
    smoothing_window: int = 3
    smoothing_method: str = "mean"
    zscore_clip: float = 3.0
    anomaly_iqr_multiplier: float = 1.5
    min_points: int = 3
    zero_guard: float = 1e-9


def metric_statistics_config(metric_name: str | None) -> MetricStatisticsConfig:
    defaults = _STATISTICS_DEFAULTS
    metric_overrides = _as_dict(_STATISTICS_METRICS.get(_normalize(metric_name)))

    smoothing_window = _as_int(
        metric_overrides.get("smoothing_window", defaults.get("smoothing_window", 3)),
        3,
    )
    smoothing_method = _normalize(
        metric_overrides.get("smoothing_method", defaults.get("smoothing_method", "mean"))
    )
    if smoothing_method not in {"mean", "median"}:
        smoothing_method = "mean"
    zscore_clip = max(
        0.5,
        _as_float(metric_overrides.get("zscore_clip", defaults.get("zscore_clip", 3.0)), 3.0),
    )
    anomaly_iqr_multiplier = max(
        0.5,
        _as_float(
            metric_overrides.get(
                "anomaly_iqr_multiplier",
                defaults.get("anomaly_iqr_multiplier", 1.5),
            ),
            1.5,
        ),
    )
    min_points = _as_int(
        metric_overrides.get("min_points", defaults.get("min_points", 3)),
        3,
    )
    zero_guard = max(
        1e-12,
        _as_float(metric_overrides.get("zero_guard", defaults.get("zero_guard", 1e-9)), 1e-9),
    )

    return MetricStatisticsConfig(
        smoothing_window=smoothing_window,
        smoothing_method=smoothing_method,
        zscore_clip=zscore_clip,
        anomaly_iqr_multiplier=anomaly_iqr_multiplier,
        min_points=min_points,
        zero_guard=zero_guard,
    )


def coerce_numeric_series(values: Sequence[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            out.append(parsed)
    return out


def zscore_normalize(
    values: Sequence[Any],
    *,
    clip_abs: float = 3.0,
    zero_guard: float = 1e-9,
) -> list[float]:
    series = coerce_numeric_series(values)
    if not series:
        return []
    mu = mean(series)
    sigma = pstdev(series) if len(series) >= 2 else 0.0
    denom = max(abs(sigma), max(zero_guard, 1e-12))

    normalized: list[float] = []
    for value in series:
        z = (value - mu) / denom
        clipped = max(-clip_abs, min(clip_abs, z))
        normalized.append(float(round(clipped, 6)))
    return normalized


def rolling_mean(
    values: Sequence[Any],
    *,
    window: int,
) -> list[float]:
    series = coerce_numeric_series(values)
    if not series:
        return []
    w = max(1, int(window))
    out: list[float] = []
    for idx in range(len(series)):
        start = max(0, idx - w + 1)
        bucket = series[start : idx + 1]
        out.append(round(mean(bucket), 6))
    return out


def rolling_median(
    values: Sequence[Any],
    *,
    window: int,
) -> list[float]:
    series = coerce_numeric_series(values)
    if not series:
        return []
    w = max(1, int(window))
    out: list[float] = []
    for idx in range(len(series)):
        start = max(0, idx - w + 1)
        bucket = series[start : idx + 1]
        out.append(round(float(median(bucket)), 6))
    return out

