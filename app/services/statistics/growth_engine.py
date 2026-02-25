from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from app.services.statistics.normalization import coerce_numeric_series


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


_GROWTH_RULES = _as_dict(_load_business_rules().get("growth_engine"))
_GROWTH_DEFAULTS = _as_dict(_GROWTH_RULES.get("defaults"))
_GROWTH_METRICS = _as_dict(_GROWTH_RULES.get("metrics"))


@dataclass(frozen=True)
class GrowthEngineConfig:
    short_window: int = 3
    mid_window: int = 6
    long_window: int = 12
    cagr_periods: int = 12

    min_short_history: int = 4
    min_mid_history: int = 7
    min_long_history: int = 13
    min_cagr_history: int = 13
    min_acceleration_history: int = 7
    zero_guard: float = 1e-9


def metric_growth_config(metric_name: str | None) -> GrowthEngineConfig:
    defaults = _GROWTH_DEFAULTS
    overrides = _as_dict(_GROWTH_METRICS.get(_normalize(metric_name)))

    short_window = _as_int(overrides.get("short_window", defaults.get("short_window", 3)), 3)
    mid_window = _as_int(overrides.get("mid_window", defaults.get("mid_window", 6)), 6)
    long_window = _as_int(overrides.get("long_window", defaults.get("long_window", 12)), 12)
    cagr_periods = _as_int(overrides.get("cagr_periods", defaults.get("cagr_periods", long_window)), long_window)

    min_short_history = _as_int(
        overrides.get("min_short_history", defaults.get("min_short_history", short_window + 1)),
        short_window + 1,
    )
    min_mid_history = _as_int(
        overrides.get("min_mid_history", defaults.get("min_mid_history", mid_window + 1)),
        mid_window + 1,
    )
    min_long_history = _as_int(
        overrides.get("min_long_history", defaults.get("min_long_history", long_window + 1)),
        long_window + 1,
    )
    min_cagr_history = _as_int(
        overrides.get("min_cagr_history", defaults.get("min_cagr_history", cagr_periods + 1)),
        cagr_periods + 1,
    )
    min_acceleration_history = _as_int(
        overrides.get(
            "min_acceleration_history",
            defaults.get("min_acceleration_history", mid_window + 1),
        ),
        mid_window + 1,
    )

    zero_guard = max(
        1e-12,
        _as_float(overrides.get("zero_guard", defaults.get("zero_guard", 1e-9)), 1e-9),
    )

    return GrowthEngineConfig(
        short_window=short_window,
        mid_window=mid_window,
        long_window=long_window,
        cagr_periods=cagr_periods,
        min_short_history=min_short_history,
        min_mid_history=min_mid_history,
        min_long_history=min_long_history,
        min_cagr_history=min_cagr_history,
        min_acceleration_history=min_acceleration_history,
        zero_guard=zero_guard,
    )


def compute_growth_signals(
    values: Sequence[Any],
    *,
    metric_name: str | None = None,
    config: GrowthEngineConfig | None = None,
) -> dict[str, Any]:
    cfg = config or metric_growth_config(metric_name)
    series = coerce_numeric_series(values)
    points = len(series)

    short_rate = _moving_growth(series, window=cfg.short_window, zero_guard=cfg.zero_guard)
    mid_rate = _moving_growth(series, window=cfg.mid_window, zero_guard=cfg.zero_guard)
    long_rate = _moving_growth(series, window=cfg.long_window, zero_guard=cfg.zero_guard)
    cagr_rate = _cagr(series, periods=cfg.cagr_periods, zero_guard=cfg.zero_guard)

    acceleration = _acceleration(short_rate=short_rate, mid_rate=mid_rate, long_rate=long_rate)
    insufficient = {
        "short": points < cfg.min_short_history,
        "mid": points < cfg.min_mid_history,
        "long": points < cfg.min_long_history,
        "cagr": points < cfg.min_cagr_history,
        "acceleration": points < cfg.min_acceleration_history,
    }

    warnings: list[str] = []
    for key, is_short in insufficient.items():
        if is_short:
            warnings.append(f"Insufficient history for {key} horizon growth signal.")

    valid_count = sum(0 if flag else 1 for flag in insufficient.values())
    confidence = max(0.2, round(valid_count / max(1, len(insufficient)), 6))

    return {
        "status": "partial" if any(insufficient.values()) else "success",
        "metric_name": metric_name,
        "points_used": points,
        "warnings": warnings,
        "confidence_score": confidence,
        "windows": {
            "short": cfg.short_window,
            "mid": cfg.mid_window,
            "long": cfg.long_window,
            "cagr_periods": cfg.cagr_periods,
        },
        "minimum_history": {
            "short": cfg.min_short_history,
            "mid": cfg.min_mid_history,
            "long": cfg.min_long_history,
            "cagr": cfg.min_cagr_history,
            "acceleration": cfg.min_acceleration_history,
        },
        "insufficient_history": insufficient,
        "moving_growth_rates": {
            "short": _round_or_none(short_rate),
            "mid": _round_or_none(mid_rate),
            "long": _round_or_none(long_rate),
        },
        "cagr": {
            "rate": _round_or_none(cagr_rate),
            "periods": cfg.cagr_periods,
            "insufficient_history": insufficient["cagr"],
        },
        "acceleration_metrics": {
            "short_to_mid": _round_or_none(acceleration["short_to_mid"]),
            "mid_to_long": _round_or_none(acceleration["mid_to_long"]),
            "trend_acceleration": _round_or_none(acceleration["trend_acceleration"]),
            "insufficient_history": insufficient["acceleration"],
        },
    }


def compute_growth_context(
    metric_series: Mapping[str, Sequence[Any]],
    *,
    preferred_metric_candidates: Sequence[str] = (),
) -> dict[str, Any]:
    metrics_payload: dict[str, Any] = {}
    partial_count = 0
    warnings: list[str] = []

    for metric_name in sorted(metric_series):
        values = metric_series.get(metric_name) or []
        payload = compute_growth_signals(values, metric_name=metric_name)
        metrics_payload[metric_name] = payload
        if payload.get("status") == "partial":
            partial_count += 1
            for warning in payload.get("warnings", []):
                warnings.append(f"{metric_name}: {warning}")

    primary_metric = _pick_primary_metric(metrics_payload, preferred_metric_candidates)
    primary_payload = metrics_payload.get(primary_metric, {}) if primary_metric else {}
    horizons = _primary_horizons(primary_payload)

    confidence_values = [
        float(payload.get("confidence_score") or 0.2)
        for payload in metrics_payload.values()
        if isinstance(payload, Mapping)
    ]
    confidence = round(mean(confidence_values), 6) if confidence_values else 0.2

    return {
        "status": "partial" if partial_count > 0 else "success",
        "confidence_score": confidence,
        "warnings": warnings,
        "primary_metric": primary_metric,
        "primary_horizons": horizons,
        "metrics": metrics_payload,
    }


def _moving_growth(
    series: Sequence[float],
    *,
    window: int,
    zero_guard: float,
) -> float | None:
    w = max(1, int(window))
    if len(series) < (w + 1):
        return None
    previous = series[-(w + 1)]
    current = series[-1]
    denominator = max(abs(previous), zero_guard)
    return (current - previous) / denominator


def _cagr(
    series: Sequence[float],
    *,
    periods: int,
    zero_guard: float,
) -> float | None:
    p = max(1, int(periods))
    if len(series) < (p + 1):
        return None
    start = series[-(p + 1)]
    end = series[-1]
    if start <= zero_guard or end <= 0.0:
        return None
    return (end / start) ** (1.0 / p) - 1.0


def _acceleration(
    *,
    short_rate: float | None,
    mid_rate: float | None,
    long_rate: float | None,
) -> dict[str, float | None]:
    short_to_mid = None
    if short_rate is not None and mid_rate is not None:
        short_to_mid = short_rate - mid_rate

    mid_to_long = None
    if mid_rate is not None and long_rate is not None:
        mid_to_long = mid_rate - long_rate

    trend_acceleration = None
    if short_rate is not None and mid_rate is not None and long_rate is not None:
        trend_acceleration = short_rate - (2.0 * mid_rate) + long_rate

    return {
        "short_to_mid": short_to_mid,
        "mid_to_long": mid_to_long,
        "trend_acceleration": trend_acceleration,
    }


def _pick_primary_metric(
    metrics_payload: Mapping[str, Any],
    candidates: Sequence[str],
) -> str | None:
    normalized_candidates = [_normalize(v) for v in candidates if _normalize(v)]
    for candidate in normalized_candidates:
        for metric_name in metrics_payload:
            if _normalize(metric_name) == candidate:
                return metric_name

    best_name: str | None = None
    best_score = -1.0
    for metric_name, payload in metrics_payload.items():
        if not isinstance(payload, Mapping):
            continue
        horizons = payload.get("moving_growth_rates")
        if not isinstance(horizons, Mapping):
            continue
        score = 0.0
        for key in ("short", "mid", "long"):
            value = horizons.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                score += 1.0
        if score > best_score:
            best_score = score
            best_name = metric_name
    return best_name


def _primary_horizons(primary_payload: Mapping[str, Any]) -> dict[str, Any]:
    moving = primary_payload.get("moving_growth_rates")
    if not isinstance(moving, Mapping):
        moving = {}
    acceleration = primary_payload.get("acceleration_metrics")
    if not isinstance(acceleration, Mapping):
        acceleration = {}
    insufficient = primary_payload.get("insufficient_history")
    if not isinstance(insufficient, Mapping):
        insufficient = {}

    return {
        "short_growth": _round_or_none(moving.get("short")),
        "mid_growth": _round_or_none(moving.get("mid")),
        "long_growth": _round_or_none(moving.get("long")),
        "cagr": _round_or_none(_as_optional_float(_as_dict(primary_payload.get("cagr")).get("rate"))),
        "trend_acceleration": _round_or_none(acceleration.get("trend_acceleration")),
        "insufficient_history": {
            "short": bool(insufficient.get("short")),
            "mid": bool(insufficient.get("mid")),
            "long": bool(insufficient.get("long")),
            "cagr": bool(insufficient.get("cagr")),
            "acceleration": bool(insufficient.get("acceleration")),
        },
    }


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


def _round_or_none(value: Any) -> float | None:
    parsed = _as_optional_float(value)
    if parsed is None:
        return None
    return round(parsed, 6)

