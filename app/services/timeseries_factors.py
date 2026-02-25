"""
app/services/timeseries_factors.py

Deterministic time-series factor extraction with rule-driven windows and
thresholds from ``config/business_rules.yaml``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence


_BUSINESS_RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "business_rules.yaml"


@lru_cache(maxsize=1)
def _load_business_rules() -> dict[str, Any]:
    try:
        raw = _BUSINESS_RULES_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
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


_TIMESERIES_RULES = _as_dict(_load_business_rules().get("timeseries_factors"))
_WINDOW_RULES = _as_dict(_TIMESERIES_RULES.get("windows"))
_THRESHOLD_RULES = _as_dict(_TIMESERIES_RULES.get("thresholds"))


@dataclass(frozen=True)
class TimeseriesFactorConfig:
    momentum_short_window: int = _as_int(_WINDOW_RULES.get("momentum_short_window"), 3)
    momentum_long_window: int = _as_int(_WINDOW_RULES.get("momentum_long_window"), 8)
    volatility_window: int = _as_int(_WINDOW_RULES.get("volatility_window"), 6)
    structural_break_min_segment: int = _as_int(_WINDOW_RULES.get("structural_break_min_segment"), 6)
    cycle_window: int = _as_int(_WINDOW_RULES.get("cycle_window"), 8)

    momentum_up_threshold: float = _as_float(_THRESHOLD_RULES.get("momentum_up_threshold"), 0.03)
    momentum_down_threshold: float = _as_float(_THRESHOLD_RULES.get("momentum_down_threshold"), 0.03)
    volatility_high_ratio: float = _as_float(_THRESHOLD_RULES.get("volatility_high_ratio"), 1.25)
    volatility_low_ratio: float = _as_float(_THRESHOLD_RULES.get("volatility_low_ratio"), 0.80)
    structural_break_score_threshold: float = _as_float(
        _THRESHOLD_RULES.get("structural_break_score_threshold"),
        0.95,
    )
    structural_break_min_shift: float = _as_float(
        _THRESHOLD_RULES.get("structural_break_min_shift"),
        0.0,
    )
    cycle_peak_position_threshold: float = _as_float(
        _THRESHOLD_RULES.get("cycle_peak_position_threshold"),
        0.80,
    )
    cycle_trough_position_threshold: float = _as_float(
        _THRESHOLD_RULES.get("cycle_trough_position_threshold"),
        0.20,
    )
    zero_guard: float = _as_float(_THRESHOLD_RULES.get("zero_guard"), 1e-9)


def compute_timeseries_factors(
    series: Sequence[float | int | Mapping[str, Any]],
    *,
    config: TimeseriesFactorConfig | None = None,
) -> dict[str, Any]:
    """
    Compute deterministic time-series factor flags.

    Returned machine-readable flags:
      - ``momentum_up``
      - ``momentum_down``
      - ``volatility_regime`` (``high`` | ``normal`` | ``low`` | ``insufficient_history``)
      - ``structural_break_detected``
      - ``cycle_state`` (``expansion`` | ``contraction`` | ``peak`` | ``trough`` | ``neutral`` | ``insufficient_history``)
    """
    cfg = config or TimeseriesFactorConfig()
    values = _coerce_series_values(series)
    points = len(values)

    momentum_score = _momentum_score(
        values,
        short_window=cfg.momentum_short_window,
        long_window=cfg.momentum_long_window,
        zero_guard=cfg.zero_guard,
    )
    momentum_up = momentum_score >= cfg.momentum_up_threshold
    momentum_down = momentum_score <= -cfg.momentum_down_threshold

    vol_regime, vol_meta = _volatility_regime(
        values,
        window=cfg.volatility_window,
        high_ratio=cfg.volatility_high_ratio,
        low_ratio=cfg.volatility_low_ratio,
        zero_guard=cfg.zero_guard,
    )

    break_meta = _structural_break_detection(
        values,
        min_segment=cfg.structural_break_min_segment,
        score_threshold=cfg.structural_break_score_threshold,
        min_shift=cfg.structural_break_min_shift,
        zero_guard=cfg.zero_guard,
    )

    cycle_state, cycle_meta = _cycle_state(
        values,
        momentum_up=momentum_up,
        momentum_down=momentum_down,
        window=cfg.cycle_window,
        peak_position=cfg.cycle_peak_position_threshold,
        trough_position=cfg.cycle_trough_position_threshold,
        zero_guard=cfg.zero_guard,
    )

    return {
        "momentum_up": momentum_up,
        "momentum_down": momentum_down,
        "volatility_regime": vol_regime,
        "structural_break_detected": break_meta["detected"],
        "cycle_state": cycle_state,
        "diagnostics": {
            "points_used": points,
            "momentum_score": round(momentum_score, 6),
            "volatility": vol_meta,
            "structural_break": break_meta,
            "cycle": cycle_meta,
        },
        "applied_rules": {
            "windows": {
                "momentum_short_window": cfg.momentum_short_window,
                "momentum_long_window": cfg.momentum_long_window,
                "volatility_window": cfg.volatility_window,
                "structural_break_min_segment": cfg.structural_break_min_segment,
                "cycle_window": cfg.cycle_window,
            },
            "thresholds": {
                "momentum_up_threshold": cfg.momentum_up_threshold,
                "momentum_down_threshold": cfg.momentum_down_threshold,
                "volatility_high_ratio": cfg.volatility_high_ratio,
                "volatility_low_ratio": cfg.volatility_low_ratio,
                "structural_break_score_threshold": cfg.structural_break_score_threshold,
                "structural_break_min_shift": cfg.structural_break_min_shift,
                "cycle_peak_position_threshold": cfg.cycle_peak_position_threshold,
                "cycle_trough_position_threshold": cfg.cycle_trough_position_threshold,
            },
        },
    }


class TimeseriesFactorsEngine:
    """Thin deterministic wrapper around :func:`compute_timeseries_factors`."""

    def __init__(self, *, config: TimeseriesFactorConfig | None = None) -> None:
        self._config = config or TimeseriesFactorConfig()

    def evaluate(self, series: Sequence[float | int | Mapping[str, Any]]) -> dict[str, Any]:
        return compute_timeseries_factors(series, config=self._config)


def _coerce_series_values(series: Sequence[float | int | Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    for point in series:
        candidate: Any
        if isinstance(point, Mapping):
            if "value" in point:
                candidate = point.get("value")
            elif "metric_value" in point:
                candidate = point.get("metric_value")
            else:
                continue
        else:
            candidate = point

        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _momentum_score(
    values: Sequence[float],
    *,
    short_window: int,
    long_window: int,
    zero_guard: float,
) -> float:
    if len(values) < max(short_window, long_window):
        return 0.0
    short_avg = mean(values[-short_window:])
    long_avg = mean(values[-long_window:])
    return (short_avg - long_avg) / max(abs(long_avg), zero_guard)


def _volatility_regime(
    values: Sequence[float],
    *,
    window: int,
    high_ratio: float,
    low_ratio: float,
    zero_guard: float,
) -> tuple[str, dict[str, Any]]:
    returns = _returns(values, zero_guard=zero_guard)
    if len(returns) < 2:
        return "insufficient_history", {
            "recent_volatility": None,
            "baseline_volatility": None,
            "volatility_ratio": None,
            "window": window,
        }

    w = min(window, len(returns))
    recent = returns[-w:]
    baseline = returns[:-w]
    if len(baseline) < 2:
        baseline = returns

    recent_vol = pstdev(recent) if len(recent) >= 2 else 0.0
    baseline_vol = pstdev(baseline) if len(baseline) >= 2 else recent_vol
    ratio = recent_vol / max(baseline_vol, zero_guard)

    if ratio >= high_ratio:
        regime = "high"
    elif ratio <= low_ratio:
        regime = "low"
    else:
        regime = "normal"

    return regime, {
        "recent_volatility": round(recent_vol, 6),
        "baseline_volatility": round(baseline_vol, 6),
        "volatility_ratio": round(ratio, 6),
        "window": w,
    }


def _returns(values: Sequence[float], *, zero_guard: float) -> list[float]:
    if len(values) < 2:
        return []
    out: list[float] = []
    for previous, current in zip(values[:-1], values[1:]):
        out.append((current - previous) / max(abs(previous), zero_guard))
    return out


def _linear_detrended(values: Sequence[float]) -> list[float]:
    n = len(values)
    if n <= 1:
        return list(values)

    x_mean = (n - 1) / 2.0
    y_mean = mean(values)
    numerator = 0.0
    denominator = 0.0
    for idx, val in enumerate(values):
        x_centered = idx - x_mean
        numerator += x_centered * (val - y_mean)
        denominator += x_centered * x_centered
    slope = numerator / denominator if denominator != 0.0 else 0.0
    intercept = y_mean - slope * x_mean

    return [val - (intercept + slope * idx) for idx, val in enumerate(values)]


def _structural_break_detection(
    values: Sequence[float],
    *,
    min_segment: int,
    score_threshold: float,
    min_shift: float,
    zero_guard: float,
) -> dict[str, Any]:
    n = len(values)
    if n < (min_segment * 2):
        return {
            "detected": False,
            "score": 0.0,
            "split_index": None,
            "level_shift": 0.0,
            "min_segment": min_segment,
        }

    residuals = _linear_detrended(values)
    best_score = 0.0
    best_split: int | None = None
    best_shift = 0.0

    for split in range(min_segment, n - min_segment + 1):
        left_raw = values[:split]
        right_raw = values[split:]
        left = residuals[:split]
        right = residuals[split:]
        if len(left) < min_segment or len(right) < min_segment:
            continue

        left_mean = mean(left)
        right_mean = mean(right)
        diff = abs(right_mean - left_mean)
        sigma = pstdev(left + right) if len(left) + len(right) >= 2 else 0.0
        score = diff / max(sigma, zero_guard)
        level_shift = abs(mean(right_raw) - mean(left_raw))

        if score > best_score:
            best_score = score
            best_split = split
            best_shift = level_shift

    detected = (best_score >= score_threshold) and (best_shift >= min_shift)
    return {
        "detected": bool(detected),
        "score": round(best_score, 6),
        "split_index": best_split,
        "level_shift": round(best_shift, 6),
        "min_segment": min_segment,
    }


def _cycle_state(
    values: Sequence[float],
    *,
    momentum_up: bool,
    momentum_down: bool,
    window: int,
    peak_position: float,
    trough_position: float,
    zero_guard: float,
) -> tuple[str, dict[str, Any]]:
    if len(values) < window:
        return "insufficient_history", {
            "position": None,
            "window": window,
        }

    segment = list(values[-window:])
    minimum = min(segment)
    maximum = max(segment)
    latest = segment[-1]
    position = (latest - minimum) / max(maximum - minimum, zero_guard)

    if momentum_up:
        state = "expansion"
    elif momentum_down:
        state = "contraction"
    elif position >= peak_position:
        state = "peak"
    elif position <= trough_position:
        state = "trough"
    else:
        state = "neutral"

    return state, {
        "position": round(position, 6),
        "window": window,
        "latest": round(latest, 6),
        "minimum": round(minimum, 6),
        "maximum": round(maximum, 6),
    }
