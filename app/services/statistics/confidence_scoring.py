"""
app/services/statistics/confidence_scoring.py

Unified deterministic confidence scoring model for analytical insights.

This module replaces ad-hoc per-node confidence formulas with a composable
model that systematically evaluates four dimensions:

    1. **Depth**       — number of data points vs minimum thresholds
    2. **Volatility**  — coefficient of variation and regime detection
    3. **Anomaly**     — presence and density of IQR-flagged outliers
    4. **Consistency**  — agreement between correlated signals

Each dimension produces a component score in [0, 1].  The final confidence
is the geometric mean of the components, clamped to [floor, ceiling] and
optionally capped by a tier.

Formula
-------
    confidence = clamp(
        (depth^w_d × volatility^w_v × anomaly^w_a × consistency^w_c) ^ (1/W),
        floor, ceiling
    )
    where W = w_d + w_v + w_a + w_c

The geometric mean is used instead of arithmetic mean because it is
*multiplicatively conservative*: a single zero-score component drives the
entire confidence toward zero, which is the correct behaviour when one
dimension is catastrophically bad.

All math uses only the Python standard library.  No sklearn, numpy, or
pandas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Sequence

from app.services.statistics.normalization import coerce_numeric_series


_ZERO_GUARD = 1e-9


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConfidenceConfig:
    """Tunable parameters for confidence scoring."""

    # Component weights (relative, not required to sum to 1)
    weight_depth: float = 1.0
    weight_volatility: float = 1.0
    weight_anomaly: float = 1.0
    weight_consistency: float = 1.0

    # Depth component
    min_points: int = 3           # below this → depth_score = 0
    saturation_points: int = 12   # at this → depth_score = 1.0

    # Volatility component
    cov_ceiling: float = 2.0     # CoV above this → min volatility score
    volatility_floor: float = 0.3  # minimum volatility component score

    # Anomaly component
    anomaly_penalty_per_point: float = 0.08  # per anomalous point
    anomaly_density_cap: float = 0.50  # max total anomaly penalty
    anomaly_iqr_multiplier: float = 1.5

    # Consistency component
    min_signals_for_consistency: int = 2  # need ≥2 signals to measure agreement
    disagreement_penalty: float = 0.15    # penalty per directional conflict

    # Global bounds
    confidence_floor: float = 0.0
    confidence_ceiling: float = 1.0


# ── Component scorers ────────────────────────────────────────────────────────

def depth_score(
    n: int,
    *,
    min_points: int = 3,
    saturation_points: int = 12,
) -> dict[str, Any]:
    """
    Score based on number of data points.

    Uses logarithmic saturation:  score = min(1, ln(n) / ln(saturation))

    Returns 0 when n < min_points (hard gate).

    Returns
    -------
    dict
        component, score, n, min_points, saturation_points, sufficient
    """
    if n < max(1, min_points):
        return {
            "component": "depth",
            "score": 0.0,
            "n": n,
            "min_points": min_points,
            "saturation_points": saturation_points,
            "sufficient": False,
        }

    sat = max(2, saturation_points)
    raw = min(1.0, math.log(max(1, n)) / math.log(sat))
    return {
        "component": "depth",
        "score": round(max(0.0, raw), 6),
        "n": n,
        "min_points": min_points,
        "saturation_points": saturation_points,
        "sufficient": True,
    }


def volatility_score(
    values: Sequence[Any],
    *,
    cov_ceiling: float = 2.0,
    volatility_floor: float = 0.3,
) -> dict[str, Any]:
    """
    Score based on coefficient of variation of *detrended* residuals.

    Detrending removes the linear trend so that a monotonic rise/decline is
    not penalised as "volatile".  Only the noise around the trend matters.

    Low CoV → high score (stable series).
    High CoV → score approaches volatility_floor.

    Formula:
        residuals = values - linear_trend
        CoV = σ(residuals) / max(|μ(values)|, ε)
        score = max(floor, 1.0 - CoV / ceiling)

    Returns
    -------
    dict
        component, score, coefficient_of_variation, regime, series_length
    """
    series = coerce_numeric_series(values)

    if len(series) < 2:
        return {
            "component": "volatility",
            "score": volatility_floor,
            "coefficient_of_variation": None,
            "regime": "insufficient_history",
            "series_length": len(series),
        }

    # Detrend to isolate noise from trend
    residuals = _linear_detrend(series)
    mu_original = mean(series)
    sigma_residual = pstdev(residuals)
    cov = sigma_residual / max(abs(mu_original), _ZERO_GUARD)

    # Regime classification
    if cov > 1.0:
        regime = "extreme"
    elif cov > 0.5:
        regime = "high"
    elif cov > 0.15:
        regime = "moderate"
    else:
        regime = "low"

    ceiling = max(0.1, cov_ceiling)
    raw = 1.0 - (cov / ceiling)
    score = max(volatility_floor, min(1.0, raw))

    return {
        "component": "volatility",
        "score": round(score, 6),
        "coefficient_of_variation": round(cov, 6),
        "regime": regime,
        "series_length": len(series),
    }


def anomaly_score(
    values: Sequence[Any],
    *,
    iqr_multiplier: float = 1.5,
    penalty_per_point: float = 0.08,
    density_cap: float = 0.50,
) -> dict[str, Any]:
    """
    Score based on IQR anomaly detection.

    Each anomalous point deducts penalty_per_point from a baseline of 1.0,
    capped at density_cap total deduction.

    Formula:
        anomaly_density = anomaly_count / n
        penalty = min(density_cap, anomaly_count × penalty_per_point)
        score = 1.0 - penalty

    Returns
    -------
    dict
        component, score, anomaly_count, anomaly_density, anomaly_indexes,
        bounds, series_length
    """
    series = coerce_numeric_series(values)
    n = len(series)

    if n < 4:
        return {
            "component": "anomaly",
            "score": 1.0,
            "anomaly_count": 0,
            "anomaly_density": 0.0,
            "anomaly_indexes": [],
            "bounds": None,
            "series_length": n,
        }

    sorted_vals = sorted(series)
    q1 = _quantile(sorted_vals, 0.25)
    q3 = _quantile(sorted_vals, 0.75)
    iqr = q3 - q1
    m = max(0.5, iqr_multiplier)
    lower = q1 - m * iqr
    upper = q3 + m * iqr

    anomaly_indexes: list[int] = []
    for idx, val in enumerate(series):
        if val < lower or val > upper:
            anomaly_indexes.append(idx)

    anomaly_count = len(anomaly_indexes)
    anomaly_density = anomaly_count / max(1, n)

    penalty = min(density_cap, anomaly_count * penalty_per_point)
    score = max(0.0, 1.0 - penalty)

    return {
        "component": "anomaly",
        "score": round(score, 6),
        "anomaly_count": anomaly_count,
        "anomaly_density": round(anomaly_density, 6),
        "anomaly_indexes": anomaly_indexes,
        "bounds": {
            "q1": round(q1, 6),
            "q3": round(q3, 6),
            "iqr": round(iqr, 6),
            "lower": round(lower, 6),
            "upper": round(upper, 6),
        },
        "series_length": n,
    }


def consistency_score(
    signals: dict[str, float | None],
    *,
    min_signals: int = 2,
    disagreement_penalty: float = 0.15,
) -> dict[str, Any]:
    """
    Score based on directional agreement between correlated signals.

    Signals are classified as positive (>0), negative (<0), or neutral (=0).
    Disagreement = presence of both positive and negative directions.

    For each pair of signals that disagree directionally, the penalty is applied.

    Formula:
        conflict_count = number of (positive, negative) signal pairs
        penalty = min(0.7, conflict_count × disagreement_penalty)
        score = 1.0 - penalty

    Parameters
    ----------
    signals:
        Dict mapping signal names to directional values.
        Positive values = growth/improvement direction.
        Negative values = decline/deterioration direction.
        None values are ignored.

    Returns
    -------
    dict
        component, score, signal_count, conflict_count, directions,
        conflicts
    """
    # Filter valid signals
    valid: dict[str, float] = {}
    for name, val in signals.items():
        if val is not None:
            try:
                valid[name] = float(val)
            except (TypeError, ValueError):
                continue

    if len(valid) < min_signals:
        return {
            "component": "consistency",
            "score": 1.0,
            "signal_count": len(valid),
            "conflict_count": 0,
            "directions": {},
            "conflicts": [],
        }

    # Classify directions
    directions: dict[str, str] = {}
    for name, val in valid.items():
        if val > _ZERO_GUARD:
            directions[name] = "positive"
        elif val < -_ZERO_GUARD:
            directions[name] = "negative"
        else:
            directions[name] = "neutral"

    # Count directional conflicts (positive vs negative pairs)
    positive_names = [n for n, d in directions.items() if d == "positive"]
    negative_names = [n for n, d in directions.items() if d == "negative"]

    conflicts: list[dict[str, str]] = []
    for pos in positive_names:
        for neg in negative_names:
            conflicts.append({"positive": pos, "negative": neg})

    conflict_count = len(conflicts)
    penalty = min(0.7, conflict_count * disagreement_penalty)
    score = max(0.0, 1.0 - penalty)

    return {
        "component": "consistency",
        "score": round(score, 6),
        "signal_count": len(valid),
        "conflict_count": conflict_count,
        "directions": directions,
        "conflicts": conflicts,
    }


# ── Composite scorer ─────────────────────────────────────────────────────────

def compute_confidence(
    values: Sequence[Any],
    *,
    signals: dict[str, float | None] | None = None,
    config: ConfidenceConfig | None = None,
    tier_cap: float | None = None,
) -> dict[str, Any]:
    """
    Compute unified confidence score across all four dimensions.

    Parameters
    ----------
    values:
        Time-series data points, oldest first.
    signals:
        Optional dict of directional signal values for consistency scoring.
        Keys = signal names, values = directional magnitudes (+/- float).
    config:
        Tunable parameters. Defaults to sensible production values.
    tier_cap:
        Optional hard ceiling (e.g. 0.40 for minimal-tier forecast data).

    Returns
    -------
    dict (JSON-compatible)
        Top-level keys:
            confidence_score   – float in [0, 1]
            components         – list of per-dimension result dicts
            formula            – string description of computation
            tier_cap           – applied ceiling or None
            warnings           – list of diagnostic strings
    """
    cfg = config or ConfidenceConfig()
    series = coerce_numeric_series(values)
    n = len(series)

    # Compute individual components
    d = depth_score(n, min_points=cfg.min_points, saturation_points=cfg.saturation_points)
    v = volatility_score(series, cov_ceiling=cfg.cov_ceiling, volatility_floor=cfg.volatility_floor)
    a = anomaly_score(
        series,
        iqr_multiplier=cfg.anomaly_iqr_multiplier,
        penalty_per_point=cfg.anomaly_penalty_per_point,
        density_cap=cfg.anomaly_density_cap,
    )
    c = consistency_score(
        signals or {},
        min_signals=cfg.min_signals_for_consistency,
        disagreement_penalty=cfg.disagreement_penalty,
    )

    components = [d, v, a, c]
    warnings: list[str] = []

    # Hard gate: if depth is zero, overall confidence is zero
    if d["score"] == 0.0:
        warnings.append(
            f"Insufficient data: {n} points below minimum {cfg.min_points}."
        )
        return {
            "confidence_score": 0.0,
            "components": components,
            "formula": "depth_gate_failed",
            "tier_cap": tier_cap,
            "warnings": warnings,
        }

    # Weighted geometric mean
    pairs = [
        (d["score"], cfg.weight_depth),
        (v["score"], cfg.weight_volatility),
        (a["score"], cfg.weight_anomaly),
        (c["score"], cfg.weight_consistency),
    ]

    total_weight = sum(w for _, w in pairs)
    if total_weight < _ZERO_GUARD:
        raw_confidence = 0.0
    else:
        # Geometric mean: exp( Σ(w_i × ln(s_i)) / W )
        # Guard against log(0) by using max(score, ZERO_GUARD)
        log_sum = sum(
            w * math.log(max(s, _ZERO_GUARD))
            for s, w in pairs
        )
        raw_confidence = math.exp(log_sum / total_weight)

    # Apply tier cap
    ceiling = cfg.confidence_ceiling
    if tier_cap is not None:
        ceiling = min(ceiling, tier_cap)

    confidence = max(cfg.confidence_floor, min(ceiling, raw_confidence))

    # Generate warnings
    if v["score"] < 0.5:
        cov = v.get("coefficient_of_variation")
        warnings.append(
            f"High volatility (CoV={cov}); confidence penalised."
        )
    if a["anomaly_count"] > 0:
        warnings.append(
            f"{a['anomaly_count']} anomalous point(s) detected; "
            f"anomaly density {a['anomaly_density']:.1%}."
        )
    if c["conflict_count"] > 0:
        warnings.append(
            f"{c['conflict_count']} signal conflict(s) detected; "
            f"consistency reduced."
        )
    if tier_cap is not None and raw_confidence > tier_cap:
        warnings.append(
            f"Confidence capped from {raw_confidence:.3f} to {tier_cap} by tier."
        )

    formula = (
        f"geometric_mean(depth={d['score']:.3f}^{cfg.weight_depth}, "
        f"volatility={v['score']:.3f}^{cfg.weight_volatility}, "
        f"anomaly={a['score']:.3f}^{cfg.weight_anomaly}, "
        f"consistency={c['score']:.3f}^{cfg.weight_consistency})"
    )

    return {
        "confidence_score": round(confidence, 6),
        "raw_confidence": round(raw_confidence, 6),
        "components": components,
        "formula": formula,
        "tier_cap": tier_cap,
        "warnings": warnings,
    }


# ── Signal pattern classifiers ───────────────────────────────────────────────

def classify_signal_pattern(
    values: Sequence[Any],
    *,
    signals: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    """
    Classify the signal pattern and return confidence with a human-readable label.

    Labels:
        stable_growth       – low volatility, no anomalies, consistent positive signals
        volatile_growth     – high volatility, positive trend, possible anomalies
        sudden_spike        – anomaly detected in recent positions
        conflicting_indicators – signals disagree directionally
        stable_decline      – low volatility, consistent negative signals
        volatile_decline    – high volatility, negative trend
        insufficient_data   – too few data points
        neutral             – no strong pattern

    Returns
    -------
    dict
        pattern, confidence_score, description, components
    """
    result = compute_confidence(values, signals=signals)
    components = {c["component"]: c for c in result["components"]}

    d = components["depth"]
    v = components["volatility"]
    a = components["anomaly"]
    c = components["consistency"]

    series = coerce_numeric_series(values)
    n = len(series)

    if n < 3:
        return {
            "pattern": "insufficient_data",
            "confidence_score": 0.0,
            "description": "Too few data points for pattern classification.",
            **result,
        }

    # Determine trend direction from last vs first third
    third = max(1, n // 3)
    early_mean = mean(series[:third])
    late_mean = mean(series[-third:])
    trend_direction = "positive" if late_mean > early_mean else (
        "negative" if late_mean < early_mean else "neutral"
    )

    # Check for recent spike (anomaly in last 20% of series)
    recent_cutoff = max(1, int(n * 0.8))
    recent_anomalies = [i for i in a.get("anomaly_indexes", []) if i >= recent_cutoff]
    has_recent_spike = len(recent_anomalies) > 0

    # Pattern classification
    is_volatile = v.get("regime") in ("high", "extreme", "moderate") or v["score"] < 0.85
    has_conflicts = c["conflict_count"] > 0
    has_anomalies = a["anomaly_count"] > 0

    if has_conflicts:
        pattern = "conflicting_indicators"
        description = (
            f"{c['conflict_count']} directional conflict(s) between signals. "
            f"Indicators disagree on direction."
        )
    elif has_recent_spike:
        pattern = "sudden_spike"
        description = (
            f"Anomaly detected at position(s) {recent_anomalies} "
            f"(recent {100 - int(recent_cutoff / n * 100)}% of series)."
        )
    elif trend_direction == "positive" and not is_volatile:
        pattern = "stable_growth"
        description = "Consistent upward trend with low volatility."
    elif trend_direction == "positive" and is_volatile:
        pattern = "volatile_growth"
        description = "Upward trend present but high volatility reduces reliability."
    elif trend_direction == "negative" and not is_volatile:
        pattern = "stable_decline"
        description = "Consistent downward trend with low volatility."
    elif trend_direction == "negative" and is_volatile:
        pattern = "volatile_decline"
        description = "Downward trend with high volatility."
    else:
        pattern = "neutral"
        description = "No strong directional pattern detected."

    return {
        "pattern": pattern,
        "confidence_score": result["confidence_score"],
        "description": description,
        "trend_direction": trend_direction,
        **result,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _linear_detrend(values: list[float]) -> list[float]:
    """Remove linear trend via OLS, returning residuals."""
    n = len(values)
    if n <= 1:
        return list(values)
    x_mean = (n - 1) / 2.0
    y_mean = mean(values)
    numerator = 0.0
    denominator = 0.0
    for idx, val in enumerate(values):
        x_c = idx - x_mean
        numerator += x_c * (val - y_mean)
        denominator += x_c * x_c
    slope = numerator / denominator if denominator > _ZERO_GUARD else 0.0
    intercept = y_mean - slope * x_mean
    return [val - (intercept + slope * idx) for idx, val in enumerate(values)]


def _quantile(sorted_values: Sequence[float], p: float) -> float:
    """Linear interpolation quantile on pre-sorted values."""
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
