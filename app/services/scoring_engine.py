"""
app/services/scoring_engine.py

Relative scoring engine — compares client metrics against benchmark
distributions to produce percentile rank, z-score, deviation %, and a
normalised 0–100 composite score.

All computations are deterministic, NumPy-vectorised, and LLM-free.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class RelativeScore(BaseModel):
    """Immutable result of scoring a single client metric against benchmarks."""

    model_config = ConfigDict(frozen=True)

    metric_name: str
    client_value: float
    benchmark_mean: float
    benchmark_median: float
    benchmark_std: float

    percentile_rank: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description=(
            "Position of the client value in the benchmark distribution, "
            "expressed as a percentage (0–100). 50 = exactly at median."
        ),
    )
    z_score: float = Field(
        ...,
        description=(
            "Number of standard deviations the client value lies from the "
            "benchmark mean. Clipped to [-3, +3]."
        ),
    )
    deviation_pct: float = Field(
        ...,
        description=(
            "Percentage deviation of the client value from the benchmark "
            "median. Positive means above median."
        ),
    )
    normalised_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description=(
            "Composite score mapped to 0–100 via z-score sigmoid. "
            "50 = exactly on par with benchmark mean."
        ),
    )
    classification: str = Field(
        ...,
        description="Human-readable tier: top, above_average, average, below_average, bottom.",
    )
    nominal_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Score from nominal (non-macro-adjusted) values.",
    )
    real_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Score from inflation-adjusted real values.",
    )
    macro_resilience: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Stability of score under macro normalization (higher is more resilient).",
    )
    delta_due_to_macro: float = Field(
        ...,
        description="Difference between real_score and nominal_score.",
    )
    macro_adjustment_applied: bool = Field(
        ...,
        description="Whether macro normalization was applied to this score.",
    )
    client_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence assigned to the client-side metric observation.",
    )
    benchmark_confidence_mean: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Mean confidence across benchmark observations used for scoring.",
    )
    effective_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Combined confidence used for optional score attenuation.",
    )
    confidence_adjusted_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Confidence-weighted score used when confidence weighting is enabled.",
    )
    confidence_weighting_applied: bool = Field(
        default=False,
        description="Whether confidence weighting was applied to this score.",
    )


class CompetitiveBenchmarkMetrics(BaseModel):
    """Deterministic market-relative benchmarking metrics."""

    model_config = ConfigDict(frozen=True)

    relative_growth_index: float | None = Field(
        default=None,
        description="Entity growth index relative to market average growth.",
    )
    market_share_proxy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Proxy market share = entity MRR / (entity MRR + peer MRR total).",
    )
    stability_score: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Volatility-inverse stability score (higher is more stable).",
    )
    momentum_classification: Literal["Leader", "Challenger", "Stable", "Declining"] = Field(
        default="Stable",
        description="Deterministic momentum bucket derived from growth, share, stability, and risk.",
    )
    risk_divergence_score: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Relative risk score vs market risk (50 = parity, >50 = lower risk than market).",
    )
    explainability: dict[str, Any] = Field(
        default_factory=dict,
        description="Formula inputs, thresholds, and intermediate values used for reproducibility.",
    )


class CompositeScore(BaseModel):
    """Weighted composite competitive score across metric categories."""

    model_config = ConfigDict(frozen=True)

    overall_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Weighted average of category scores, 0–100.",
    )
    category_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Mean normalised score per category (e.g. growth, retention).",
    )
    weakest_metric: str | None = Field(
        default=None,
        description="Metric name with the lowest normalised score.",
    )
    strongest_metric: str | None = Field(
        default=None,
        description="Metric name with the highest normalised score.",
    )
    metric_details: dict[str, RelativeScore] = Field(
        default_factory=dict,
        description="Full per-metric RelativeScore breakdown.",
    )
    base_overall_score: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Overall score before confidence weighting (when applied).",
    )
    growth_score: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Momentum component used by executive composite scoring.",
    )
    level_score: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Magnitude component used by executive composite scoring.",
    )
    stability_score: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Consistency component used by executive composite scoring.",
    )
    confidence_score: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Reliability component used by executive composite scoring.",
    )
    executive_formula_applied: bool = Field(
        default=True,
        description="Whether overall_score uses the executive component-weighted formula.",
    )
    confidence_weighting_applied: bool = Field(
        default=False,
        description="Whether overall score was computed from confidence-adjusted metric scores.",
    )
    competitive_metrics: CompetitiveBenchmarkMetrics = Field(
        default_factory=CompetitiveBenchmarkMetrics,
        description="Deterministic market-relative competitive metrics for executive benchmarking.",
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ZSCORE_CLIP = 3.0
_ZERO_GUARD = 1e-9
_BUSINESS_RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "business_rules.yaml"

# Composite weights for the normalised score.
# The z-score sigmoid already encodes both percentile and deviation
# information, so we use it as the sole basis for the 0–100 mapping.

# Classification thresholds (on the normalised 0–100 scale)
_TIER_THRESHOLDS: list[tuple[float, str]] = [
    (80.0, "top"),
    (60.0, "above_average"),
    (40.0, "average"),
    (20.0, "below_average"),
]
_TIER_DEFAULT = "bottom"


@dataclass(frozen=True)
class ScoringConfig:
    use_macro_adjustment: bool = False
    use_confidence_weighting: bool = False
    macro_resilience_sensitivity: float = 1.0
    zero_guard: float = _ZERO_GUARD


@lru_cache(maxsize=1)
def _load_business_rules() -> dict[str, Any]:
    try:
        raw = _BUSINESS_RULES_PATH.read_text(encoding="utf-8")
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except (OSError, TypeError, ValueError):
        return {}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_scoring_config() -> ScoringConfig:
    scoring_rules = _as_dict(_load_business_rules().get("scoring"))
    defaults = _as_dict(scoring_rules.get("defaults"))
    return ScoringConfig(
        use_macro_adjustment=_as_bool(defaults.get("use_macro_adjustment"), False),
        use_confidence_weighting=_as_bool(defaults.get("use_confidence_weighting"), False),
        macro_resilience_sensitivity=max(
            0.1,
            _as_float(defaults.get("macro_resilience_sensitivity"), 1.0),
        ),
        zero_guard=max(
            1e-12,
            _as_float(defaults.get("zero_guard"), _ZERO_GUARD),
        ),
    )


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------


def percentile_rank(
    client_value: float,
    benchmark_values: np.ndarray,
) -> float:
    """Compute the percentile rank of *client_value* within *benchmark_values*.

    Uses the *"weak"* definition:

        percentile = (# of benchmark values < client_value) / N × 100

    Returns a float in [0, 100].  An empty benchmark array returns 50.0
    (unknown → neutral).

    Formula
    -------
    P = (count(b < v) / N) × 100
    """
    if benchmark_values.size == 0:
        return 50.0
    count_below = float(np.sum(benchmark_values < client_value))
    return round(count_below / benchmark_values.size * 100.0, 4)


def zscore(
    client_value: float,
    bench_mean: float,
    bench_std: float,
    *,
    clip: float = _ZSCORE_CLIP,
    zero_guard: float = _ZERO_GUARD,
) -> float:
    """Z-score of *client_value* relative to benchmark distribution.

    Formula
    -------
    z = (client_value − μ) / max(σ, ε)

    The result is clipped to [-clip, +clip] (default ±3) to prevent
    extreme outliers from distorting downstream scores.
    """
    denom = max(abs(bench_std), zero_guard)
    z = (client_value - bench_mean) / denom
    return round(float(np.clip(z, -clip, clip)), 6)


def deviation_pct(
    client_value: float,
    bench_median: float,
    *,
    zero_guard: float = _ZERO_GUARD,
) -> float:
    """Percentage deviation of *client_value* from benchmark median.

    Formula
    -------
    dev% = ((client_value − median) / max(|median|, ε)) × 100

    Positive values mean the client is *above* the median.
    """
    denom = max(abs(bench_median), zero_guard)
    return round((client_value - bench_median) / denom * 100.0, 4)


def normalise_to_100(z: float) -> float:
    """Map a z-score to a 0–100 scale using a sigmoid transform.

    Formula
    -------
    score = 100 / (1 + e^(−z))

    Properties:
      • z =  0  → 50  (exactly on par)
      • z = +3  → ~95 (well above average)
      • z = −3  → ~5  (well below average)
      • Monotonically increasing, smoothly bounded [0, 100].
    """
    return round(100.0 / (1.0 + math.exp(-z)), 4)


def classify(score: float) -> str:
    """Classify a normalised 0–100 score into a human-readable tier.

    Thresholds
    ----------
    ≥ 80  →  top
    ≥ 60  →  above_average
    ≥ 40  →  average
    ≥ 20  →  below_average
    <  20 →  bottom
    """
    for threshold, label in _TIER_THRESHOLDS:
        if score >= threshold:
            return label
    return _TIER_DEFAULT


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _coerce_benchmark(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Convert arbitrary input to a 1-D finite-float NumPy array."""
    if isinstance(values, np.ndarray):
        arr = values.astype(float, copy=False).ravel()
    else:
        try:
            arr = np.asarray(values, dtype=float).ravel()
        except (TypeError, ValueError):
            return np.array([], dtype=float)
    return arr[np.isfinite(arr)]


def _coerce_weight_array(
    weights: float | Sequence[Any] | np.ndarray | None,
    *,
    size: int,
) -> np.ndarray:
    if size <= 0:
        return np.array([], dtype=float)
    if weights is None:
        return np.ones(size, dtype=float)

    if isinstance(weights, np.ndarray):
        arr = weights.astype(float, copy=False).ravel()
    elif isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)):
        try:
            arr = np.asarray(weights, dtype=float).ravel()
        except (TypeError, ValueError):
            arr = np.array([], dtype=float)
    else:
        try:
            scalar = float(weights)
        except (TypeError, ValueError):
            scalar = float("nan")
        arr = np.array([scalar], dtype=float)

    if arr.size == size:
        out = arr
    elif arr.size == 1:
        out = np.full(size, arr[0], dtype=float)
    else:
        return np.ones(size, dtype=float)

    # Confidence weights are clipped to [0, 1] to avoid unstable leverage.
    out = np.clip(out, 0.0, 1.0)
    return out


def _coerce_benchmark_with_weights(
    values: Sequence[Any] | np.ndarray,
    weights: float | Sequence[Any] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if isinstance(values, np.ndarray):
        arr = values.astype(float, copy=False).ravel()
    else:
        try:
            arr = np.asarray(values, dtype=float).ravel()
        except (TypeError, ValueError):
            return np.array([], dtype=float), np.array([], dtype=float), 1.0

    raw_weights = _coerce_weight_array(weights, size=arr.size)
    mask = np.isfinite(arr) & np.isfinite(raw_weights) & (raw_weights > 0.0)
    if not np.any(mask):
        return np.array([], dtype=float), np.array([], dtype=float), 1.0

    clean_values = arr[mask]
    clean_weights = raw_weights[mask]
    weight_sum = float(np.sum(clean_weights))
    if weight_sum <= 0.0:
        clean_weights = np.ones(clean_values.size, dtype=float)
        weight_sum = float(clean_values.size)

    normalized_weights = clean_weights / weight_sum
    confidence_mean = float(np.mean(clean_weights)) if clean_weights.size else 1.0
    confidence_mean = max(0.0, min(1.0, confidence_mean))
    return clean_values, normalized_weights, confidence_mean


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    if weights.size != values.size:
        return _safe_mean(values)
    return float(np.sum(values * weights))


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    if weights.size != values.size:
        return _safe_std(values)
    mu = _weighted_mean(values, weights)
    variance = float(np.sum(weights * ((values - mu) ** 2)))
    return float(math.sqrt(max(variance, 0.0)))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    if weights.size != values.size:
        return float(np.quantile(values, q))
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    idx = int(np.searchsorted(cumulative, q, side="left"))
    idx = min(max(idx, 0), sorted_values.size - 1)
    return float(sorted_values[idx])


def _weighted_percentile_rank(
    client_value: float,
    benchmark_values: np.ndarray,
    benchmark_weights: np.ndarray,
) -> float:
    if benchmark_values.size == 0:
        return 50.0
    if benchmark_weights.size != benchmark_values.size:
        return percentile_rank(client_value, benchmark_values)
    below = benchmark_weights[benchmark_values < client_value]
    pct = float(np.sum(below)) * 100.0
    return round(max(0.0, min(100.0, pct)), 4)


def _safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size > 0 else 0.0


def _safe_median(arr: np.ndarray) -> float:
    return float(np.median(arr)) if arr.size > 0 else 0.0


def _safe_std(arr: np.ndarray) -> float:
    """Population standard deviation (ddof=0)."""
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=0))


def _normalize_direction(direction: str | None) -> Literal["higher_is_better", "lower_is_better"]:
    normalized = str(direction or "").strip().lower()
    if normalized == "lower_is_better":
        return "lower_is_better"
    return "higher_is_better"


def _directional_scalar(
    value: float,
    *,
    direction: Literal["higher_is_better", "lower_is_better"],
) -> float:
    return -value if direction == "lower_is_better" else value


def _directional_array(
    values: np.ndarray,
    *,
    direction: Literal["higher_is_better", "lower_is_better"],
) -> np.ndarray:
    if direction == "lower_is_better":
        return -values
    return values


def _clamp_confidence(value: Any, default: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    return max(0.0, min(1.0, parsed))


def _normalize_rate(value: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    if 1.0 <= abs(value) <= 500.0:
        return value / 100.0
    return value


def _to_inflation_array(
    value: float | Sequence[Any] | np.ndarray | None,
    *,
    size: int,
) -> np.ndarray:
    if size <= 0:
        return np.array([], dtype=float)
    if value is None:
        return np.full(size, np.nan, dtype=float)

    if isinstance(value, np.ndarray):
        arr = value.astype(float, copy=False).ravel()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            arr = np.asarray(value, dtype=float).ravel()
        except (TypeError, ValueError):
            arr = np.array([], dtype=float)
    else:
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            scalar = float("nan")
        arr = np.array([scalar], dtype=float)

    if arr.size == size:
        out = arr
    elif arr.size == 1:
        out = np.full(size, arr[0], dtype=float)
    else:
        out = np.full(size, np.nan, dtype=float)
        limit = min(size, arr.size)
        out[:limit] = arr[:limit]

    return np.vectorize(_normalize_rate, otypes=[float])(out)


def _infer_adjustment_mode(metric_name: str) -> str:
    normalized = metric_name.strip().lower()
    growth_tokens = (
        "growth",
        "rate",
        "ratio",
        "margin",
        "churn",
        "retention",
        "cagr",
        "roas",
        "ctr",
        "conversion",
    )
    return "growth" if any(token in normalized for token in growth_tokens) else "level"


def _adjust_for_inflation(
    values: np.ndarray,
    inflation: np.ndarray,
    *,
    mode: str,
    zero_guard: float,
) -> np.ndarray:
    if values.size == 0:
        return values.astype(float, copy=True)
    adjusted = values.astype(float, copy=True)
    infl = inflation.astype(float, copy=False)
    infl = np.where(np.isfinite(infl), infl, 0.0)

    denom = 1.0 + infl
    valid = np.isfinite(adjusted) & np.isfinite(denom) & (np.abs(denom) > zero_guard)
    out = np.full(adjusted.shape, np.nan, dtype=float)
    if mode == "growth":
        out[valid] = ((1.0 + adjusted[valid]) / denom[valid]) - 1.0
    else:
        out[valid] = adjusted[valid] / denom[valid]
    return out


def _compute_macro_resilience(
    *,
    nominal_score: float,
    real_score: float,
    sensitivity: float,
) -> float:
    delta = real_score - nominal_score
    resilience = 100.0 - abs(delta) * sensitivity
    return round(float(np.clip(resilience, 0.0, 100.0)), 4)


def _score_metric_core(
    metric_name: str,
    client_value: float,
    benchmark_values: Sequence[Any] | np.ndarray,
    *,
    direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better",
    benchmark_weights: float | Sequence[Any] | np.ndarray | None = None,
) -> dict[str, float | str]:
    use_weights = benchmark_weights is not None
    if use_weights:
        bench, weights, _ = _coerce_benchmark_with_weights(benchmark_values, benchmark_weights)
        b_mean = _weighted_mean(bench, weights)
        b_median = _weighted_quantile(bench, weights, 0.5)
        b_std = _weighted_std(bench, weights)
    else:
        bench = _coerce_benchmark(benchmark_values)
        weights = np.array([], dtype=float)
        b_mean = _safe_mean(bench)
        b_median = _safe_median(bench)
        b_std = _safe_std(bench)

    scoring_bench = _directional_array(bench, direction=direction)
    scoring_client = _directional_scalar(client_value, direction=direction)
    if use_weights:
        scoring_mean = _weighted_mean(scoring_bench, weights)
        scoring_median = _weighted_quantile(scoring_bench, weights, 0.5)
        scoring_std = _weighted_std(scoring_bench, weights)
    else:
        scoring_mean = _safe_mean(scoring_bench)
        scoring_median = _safe_median(scoring_bench)
        scoring_std = _safe_std(scoring_bench)

    pct = (
        _weighted_percentile_rank(scoring_client, scoring_bench, weights)
        if use_weights
        else percentile_rank(scoring_client, scoring_bench)
    )

    if scoring_bench.size == 0 or (scoring_bench.size < 2 and scoring_std == 0.0):
        z = 0.0
        dev = 0.0
    else:
        z = zscore(scoring_client, scoring_mean, scoring_std)
        dev = deviation_pct(scoring_client, scoring_median)

    norm = normalise_to_100(z)
    tier = classify(norm)

    return {
        "metric_name": metric_name,
        "client_value": round(client_value, 6),
        "benchmark_mean": round(b_mean, 6),
        "benchmark_median": round(b_median, 6),
        "benchmark_std": round(b_std, 6),
        "percentile_rank": pct,
        "z_score": z,
        "deviation_pct": dev,
        "normalised_score": norm,
        "classification": tier,
    }


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def score_metric(
    metric_name: str,
    client_value: float,
    benchmark_values: Sequence[Any] | np.ndarray,
    *,
    direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better",
    use_confidence_weighting: bool | None = None,
    client_confidence: float | None = None,
    benchmark_confidences: float | Sequence[Any] | np.ndarray | None = None,
    use_macro_adjustment: bool | None = None,
    client_inflation_rate: float | None = None,
    benchmark_inflation_rate: float | Sequence[Any] | np.ndarray | None = None,
    adjustment_mode: str | None = None,
) -> RelativeScore:
    """Score a single client metric against a benchmark distribution.

    Parameters
    ----------
    metric_name:
        Human-readable identifier (e.g. ``"mrr"``, ``"churn_rate"``).
    client_value:
        The client's observed value for this metric.
    benchmark_values:
        Array-like of benchmark observations for the same metric.

    Returns
    -------
    RelativeScore
        Immutable result containing percentile rank, z-score,
        deviation %, normalised 0–100 score, and classification.
    """
    resolved_direction = _normalize_direction(direction)
    nominal = _score_metric_core(
        metric_name,
        client_value,
        benchmark_values,
        direction=resolved_direction,
        benchmark_weights=benchmark_confidences,
    )
    nominal_score = float(nominal["normalised_score"])
    resolved_cfg = get_scoring_config()
    apply_confidence_weighting = (
        use_confidence_weighting
        if use_confidence_weighting is not None
        else resolved_cfg.use_confidence_weighting
    )
    resolved_client_confidence = _clamp_confidence(client_confidence, default=1.0)
    _, _, benchmark_confidence_mean = _coerce_benchmark_with_weights(
        benchmark_values,
        benchmark_confidences,
    )
    apply_macro = (
        use_macro_adjustment
        if use_macro_adjustment is not None
        else resolved_cfg.use_macro_adjustment
    )
    effective_confidence = max(
        0.0,
        min(1.0, round(resolved_client_confidence * benchmark_confidence_mean, 6)),
    )

    client_rate = _normalize_rate(float(client_inflation_rate)) if client_inflation_rate is not None else float("nan")
    bench_rate_source = benchmark_inflation_rate if benchmark_inflation_rate is not None else client_inflation_rate

    if not apply_macro:
        confidence_adjusted_score = nominal_score * effective_confidence if apply_confidence_weighting else nominal_score
        return RelativeScore(
            **nominal,
            nominal_score=nominal_score,
            real_score=nominal_score,
            macro_resilience=100.0,
            delta_due_to_macro=0.0,
            macro_adjustment_applied=False,
            client_confidence=resolved_client_confidence,
            benchmark_confidence_mean=benchmark_confidence_mean,
            effective_confidence=effective_confidence,
            confidence_adjusted_score=round(float(np.clip(confidence_adjusted_score, 0.0, 100.0)), 4),
            confidence_weighting_applied=bool(apply_confidence_weighting),
        )

    bench_values = _coerce_benchmark(benchmark_values)
    mode = adjustment_mode or _infer_adjustment_mode(metric_name)

    client_arr = np.array([client_value], dtype=float)
    client_infl = _to_inflation_array(client_rate, size=1)
    bench_infl = _to_inflation_array(bench_rate_source, size=bench_values.size)

    real_client_arr = _adjust_for_inflation(
        client_arr,
        client_infl,
        mode=mode,
        zero_guard=resolved_cfg.zero_guard,
    )
    real_bench_arr = _adjust_for_inflation(
        bench_values,
        bench_infl,
        mode=mode,
        zero_guard=resolved_cfg.zero_guard,
    )

    real_client = float(real_client_arr[0]) if real_client_arr.size else float(client_value)
    if not math.isfinite(real_client):
        real_client = float(client_value)

    real = _score_metric_core(
        metric_name,
        real_client,
        real_bench_arr,
        direction=resolved_direction,
        benchmark_weights=benchmark_confidences,
    )
    real_score = float(real["normalised_score"])
    delta = round(real_score - nominal_score, 4)
    resilience = _compute_macro_resilience(
        nominal_score=nominal_score,
        real_score=real_score,
        sensitivity=resolved_cfg.macro_resilience_sensitivity,
    )
    confidence_adjusted_score = real_score * effective_confidence if apply_confidence_weighting else real_score

    return RelativeScore(
        **real,
        nominal_score=round(nominal_score, 4),
        real_score=round(real_score, 4),
        macro_resilience=resilience,
        delta_due_to_macro=delta,
        macro_adjustment_applied=True,
        client_confidence=resolved_client_confidence,
        benchmark_confidence_mean=benchmark_confidence_mean,
        effective_confidence=effective_confidence,
        confidence_adjusted_score=round(float(np.clip(confidence_adjusted_score, 0.0, 100.0)), 4),
        confidence_weighting_applied=bool(apply_confidence_weighting),
    )


def score_metrics_batch(
    client_metrics: dict[str, float],
    benchmark_data: dict[str, Sequence[Any] | np.ndarray],
    *,
    metric_directions: Mapping[str, Literal["higher_is_better", "lower_is_better"]] | None = None,
    use_confidence_weighting: bool | None = None,
    client_confidences: Mapping[str, float] | None = None,
    benchmark_confidences: Mapping[str, float | Sequence[Any] | np.ndarray] | None = None,
    use_macro_adjustment: bool | None = None,
    client_inflation_rates: Mapping[str, float] | None = None,
    benchmark_inflation_rates: Mapping[str, float | Sequence[Any] | np.ndarray] | None = None,
    adjustment_modes: Mapping[str, str] | None = None,
) -> dict[str, RelativeScore]:
    """Score multiple client metrics against their respective benchmarks.

    Parameters
    ----------
    client_metrics:
        ``{metric_name: client_value}`` mapping.
    benchmark_data:
        ``{metric_name: array_of_benchmark_values}`` mapping.
        Metrics present in *client_metrics* but absent from
        *benchmark_data* are silently skipped.

    Returns
    -------
    dict[str, RelativeScore]
        One ``RelativeScore`` per metric that has both a client value
        and benchmark data.
    """
    results: dict[str, RelativeScore] = {}
    for name, value in client_metrics.items():
        bench = benchmark_data.get(name)
        if bench is None:
            continue
        try:
            client_val = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(client_val):
            continue
        results[name] = score_metric(
            name,
            client_val,
            bench,
            direction=(
                metric_directions.get(name, "higher_is_better")
                if metric_directions
                else "higher_is_better"
            ),
            use_confidence_weighting=use_confidence_weighting,
            client_confidence=(
                client_confidences.get(name) if client_confidences else None
            ),
            benchmark_confidences=(
                benchmark_confidences.get(name) if benchmark_confidences else None
            ),
            use_macro_adjustment=use_macro_adjustment,
            client_inflation_rate=(
                client_inflation_rates.get(name) if client_inflation_rates else None
            ),
            benchmark_inflation_rate=(
                benchmark_inflation_rates.get(name) if benchmark_inflation_rates else None
            ),
            adjustment_mode=(adjustment_modes.get(name) if adjustment_modes else None),
        )
    return results


def score_metric_macro_summary(
    metric_name: str,
    client_value: float,
    benchmark_values: Sequence[Any] | np.ndarray,
    *,
    direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better",
    use_confidence_weighting: bool | None = None,
    client_confidence: float | None = None,
    benchmark_confidences: float | Sequence[Any] | np.ndarray | None = None,
    use_macro_adjustment: bool | None = None,
    client_inflation_rate: float | None = None,
    benchmark_inflation_rate: float | Sequence[Any] | np.ndarray | None = None,
    adjustment_mode: str | None = None,
) -> dict[str, float]:
    """
    Return the macro-normalized scoring envelope requested by downstream APIs.
    """

    result = score_metric(
        metric_name,
        client_value,
        benchmark_values,
        direction=direction,
        use_confidence_weighting=use_confidence_weighting,
        client_confidence=client_confidence,
        benchmark_confidences=benchmark_confidences,
        use_macro_adjustment=use_macro_adjustment,
        client_inflation_rate=client_inflation_rate,
        benchmark_inflation_rate=benchmark_inflation_rate,
        adjustment_mode=adjustment_mode,
    )
    return {
        "nominal_score": result.nominal_score,
        "real_score": result.real_score,
        "macro_resilience": result.macro_resilience,
        "delta_due_to_macro": result.delta_due_to_macro,
    }


# ---------------------------------------------------------------------------
# Composite scoring — category-based aggregation
# ---------------------------------------------------------------------------

DEFAULT_METRIC_CATEGORIES: dict[str, list[str]] = {
    "growth": [
        "growth_rate", "mrr_growth", "revenue_growth", "new_customers",
        "customer_growth", "arr_growth", "expansion_revenue",
    ],
    "retention": [
        "churn_rate", "retention_rate", "ltv", "repeat_purchase_rate",
        "customer_lifetime_value", "net_revenue_retention", "logo_retention",
    ],
    "revenue": [
        "mrr", "arr", "arpu", "aov", "gmv", "recurring_revenue",
        "revenue", "net_revenue", "total_revenue",
    ],
    "efficiency": [
        "cac", "roas", "conversion_rate", "gross_margin",
        "utilization_rate", "ctr", "cpc", "gross_margin_rate",
    ],
}

DEFAULT_CATEGORY_WEIGHTS: dict[str, float] = {
    "growth": 0.30,
    "retention": 0.25,
    "revenue": 0.25,
    "efficiency": 0.20,
}

DEFAULT_GROWTH_METRIC_CANDIDATES: tuple[str, ...] = (
    "growth_rate",
    "mrr_growth",
    "revenue_growth",
    "arr_growth",
    "customer_growth",
)

DEFAULT_MRR_METRIC_CANDIDATES: tuple[str, ...] = (
    "mrr",
    "arr",
    "recurring_revenue",
    "revenue",
    "total_revenue",
)

DEFAULT_RISK_METRIC_CANDIDATES: tuple[str, ...] = (
    "risk_score",
    "risk_index",
    "risk_rate",
    "risk_level",
    "risk",
)

DEFAULT_STABILITY_SERIES_CANDIDATES: tuple[str, ...] = (
    "mrr",
    "arr",
    "recurring_revenue",
    "revenue",
    "growth_rate",
    "mrr_growth",
    "revenue_growth",
)

_MOMENTUM_THRESHOLDS: dict[str, float] = {
    "leader_growth_index_min": 1.05,
    "leader_market_share_min": 0.20,
    "leader_stability_min": 65.0,
    "challenger_growth_index_min": 1.02,
    "stable_growth_index_min": 0.95,
    "risk_headwind_threshold": 45.0,
}


def _resolve_category_weights(
    overrides: Mapping[str, float] | None,
) -> dict[str, float]:
    """Return normalised category weights (sum to 1.0).

    Applies *overrides* on top of ``DEFAULT_CATEGORY_WEIGHTS``, clamps
    negatives to zero, and re-normalises so the total equals 1.0.
    """
    weights = dict(DEFAULT_CATEGORY_WEIGHTS)
    if overrides:
        for key, val in overrides.items():
            try:
                weights[key] = float(val)
            except (TypeError, ValueError):
                continue

    # Clamp negatives
    for key in list(weights):
        weights[key] = max(0.0, weights[key])

    total = sum(weights.values())
    if total <= 0.0:
        return {k: 1.0 / len(DEFAULT_CATEGORY_WEIGHTS) for k in DEFAULT_CATEGORY_WEIGHTS}
    for key in weights:
        weights[key] = weights[key] / total
    return weights


def _build_category_index(
    category_map: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    """Invert category → metrics mapping to metric → category lookup."""
    index: dict[str, str] = {}
    for category, metrics in category_map.items():
        for metric in metrics:
            index[metric.strip().lower()] = category
    return index


def _coerce_scalar(value: Any) -> float | None:
    if isinstance(value, Mapping):
        value = value.get("value")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _metric_lookup(metrics: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_name in metrics.keys():
        name = str(raw_name).strip()
        normalized = name.lower()
        if normalized and normalized not in out:
            out[normalized] = name
    return out


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _resolve_metric_pair(
    client_metrics: Mapping[str, Any],
    benchmark_data: Mapping[str, Sequence[Any] | np.ndarray],
    candidates: Sequence[str],
) -> tuple[str | None, float | None, np.ndarray]:
    client_index = _metric_lookup(client_metrics)
    benchmark_index = _metric_lookup(benchmark_data)
    for candidate in candidates:
        normalized = str(candidate).strip().lower()
        if not normalized:
            continue
        client_key = client_index.get(normalized)
        benchmark_key = benchmark_index.get(normalized)
        if client_key is None or benchmark_key is None:
            continue
        client_value = _coerce_scalar(client_metrics.get(client_key))
        if client_value is None:
            continue
        benchmark_values = _coerce_benchmark(benchmark_data.get(benchmark_key, []))
        if benchmark_values.size == 0:
            continue
        return client_key, client_value, benchmark_values
    return None, None, np.array([], dtype=float)


def _resolve_series(
    metric_series: Mapping[str, Sequence[float] | np.ndarray],
    candidates: Sequence[str],
) -> tuple[str | None, np.ndarray]:
    series_index = _metric_lookup(metric_series)
    for candidate in candidates:
        normalized = str(candidate).strip().lower()
        if not normalized:
            continue
        series_key = series_index.get(normalized)
        if series_key is None:
            continue
        series_values = _coerce_benchmark(metric_series.get(series_key, []))
        if series_values.size >= 2:
            return series_key, series_values
    return None, np.array([], dtype=float)


def _normalize_rate_array(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    normalized = np.vectorize(_normalize_rate, otypes=[float])(values.astype(float, copy=False))
    return normalized[np.isfinite(normalized)]


def _compute_relative_growth_index(
    *,
    client_growth: float | None,
    market_growth_values: np.ndarray,
    zero_guard: float,
) -> tuple[float | None, float | None]:
    if client_growth is None or market_growth_values.size == 0:
        return None, None
    market_avg = _safe_mean(market_growth_values)
    denominator = 1.0 + market_avg
    if not math.isfinite(denominator) or abs(denominator) <= zero_guard:
        return None, market_avg
    index = (1.0 + client_growth) / denominator
    if not math.isfinite(index):
        return None, market_avg
    return float(index), float(market_avg)


def _compute_market_share_proxy(
    *,
    client_mrr: float | None,
    peer_mrr_values: np.ndarray,
    zero_guard: float,
) -> tuple[float | None, float | None]:
    if client_mrr is None or peer_mrr_values.size == 0:
        return None, None
    client_value = max(0.0, float(client_mrr))
    peers = peer_mrr_values[np.isfinite(peer_mrr_values)]
    peers = peers[peers > 0.0]
    total_market_mrr = client_value + float(np.sum(peers))
    if total_market_mrr <= zero_guard:
        return None, None
    market_share = client_value / total_market_mrr
    return float(np.clip(market_share, 0.0, 1.0)), float(total_market_mrr)


def _volatility_inverse_score(
    values: np.ndarray,
    *,
    zero_guard: float,
) -> tuple[float, float | None]:
    series = values[np.isfinite(values)]
    if series.size < 2:
        return 50.0, None
    series_mean = float(np.mean(series))
    series_std = float(np.std(series, ddof=0))
    coefficient = series_std / max(abs(series_mean), zero_guard)
    score = 100.0 / (1.0 + coefficient)
    return float(np.clip(round(score, 2), 0.0, 100.0)), float(coefficient)


def _compute_risk_divergence_score(
    *,
    client_risk: float | None,
    market_risk_values: np.ndarray,
    zero_guard: float,
) -> tuple[float | None, float | None]:
    if client_risk is None or market_risk_values.size == 0:
        return None, None
    market_risk = _safe_mean(market_risk_values)
    if not math.isfinite(market_risk):
        return None, None

    if abs(market_risk) <= zero_guard:
        if abs(client_risk) <= zero_guard:
            return 50.0, market_risk
        return (0.0 if client_risk > 0.0 else 100.0), market_risk

    normalized_gap = (market_risk - client_risk) / abs(market_risk)
    score = 50.0 + (50.0 * normalized_gap)
    return float(np.clip(score, 0.0, 100.0)), float(market_risk)


def _classify_momentum(
    *,
    relative_growth_index: float | None,
    market_share_proxy: float | None,
    stability_score: float,
    risk_divergence_score: float | None,
) -> Literal["Leader", "Challenger", "Stable", "Declining"]:
    risk_headwind = (
        risk_divergence_score is not None
        and risk_divergence_score < _MOMENTUM_THRESHOLDS["risk_headwind_threshold"]
    )
    if relative_growth_index is None:
        return "Declining" if risk_headwind else "Stable"

    share = market_share_proxy if market_share_proxy is not None else 0.0
    if (
        relative_growth_index >= _MOMENTUM_THRESHOLDS["leader_growth_index_min"]
        and share >= _MOMENTUM_THRESHOLDS["leader_market_share_min"]
        and stability_score >= _MOMENTUM_THRESHOLDS["leader_stability_min"]
        and not risk_headwind
    ):
        return "Leader"

    if (
        relative_growth_index >= _MOMENTUM_THRESHOLDS["challenger_growth_index_min"]
        and not risk_headwind
    ):
        return "Challenger"

    if (
        relative_growth_index < _MOMENTUM_THRESHOLDS["stable_growth_index_min"]
        or risk_headwind
    ):
        return "Declining"

    return "Stable"


def _compute_competitive_benchmark_metrics(
    *,
    client_metrics: Mapping[str, Any],
    benchmark_data: Mapping[str, Sequence[Any] | np.ndarray],
    metric_scores: Mapping[str, float],
    metric_series: Mapping[str, Sequence[float] | np.ndarray] | None,
    growth_metric_candidates: Sequence[str],
    mrr_metric_candidates: Sequence[str],
    risk_metric_candidates: Sequence[str],
    stability_metric_candidates: Sequence[str],
    zero_guard: float,
) -> CompetitiveBenchmarkMetrics:
    growth_metric, raw_client_growth, raw_market_growth = _resolve_metric_pair(
        client_metrics,
        benchmark_data,
        growth_metric_candidates,
    )
    client_growth = (
        _normalize_rate(raw_client_growth)
        if raw_client_growth is not None and math.isfinite(raw_client_growth)
        else None
    )
    market_growth_values = _normalize_rate_array(raw_market_growth)
    relative_growth_index, market_growth_avg = _compute_relative_growth_index(
        client_growth=client_growth,
        market_growth_values=market_growth_values,
        zero_guard=zero_guard,
    )

    mrr_metric, client_mrr, peer_mrr_values = _resolve_metric_pair(
        client_metrics,
        benchmark_data,
        mrr_metric_candidates,
    )
    market_share_proxy, total_market_mrr = _compute_market_share_proxy(
        client_mrr=client_mrr,
        peer_mrr_values=peer_mrr_values,
        zero_guard=zero_guard,
    )

    stability_source = "cross_metric_scores"
    stability_values = np.asarray(list(metric_scores.values()), dtype=float)
    if metric_series:
        selected_series_name, selected_series = _resolve_series(
            metric_series,
            stability_metric_candidates,
        )
        if selected_series.size >= 2:
            stability_values = selected_series
            stability_source = f"metric_series:{selected_series_name}"
    stability_score, volatility_coefficient = _volatility_inverse_score(
        stability_values,
        zero_guard=zero_guard,
    )

    risk_metric, client_risk, market_risk_values = _resolve_metric_pair(
        client_metrics,
        benchmark_data,
        risk_metric_candidates,
    )
    risk_divergence_score, market_risk_avg = _compute_risk_divergence_score(
        client_risk=client_risk,
        market_risk_values=market_risk_values,
        zero_guard=zero_guard,
    )

    momentum = _classify_momentum(
        relative_growth_index=relative_growth_index,
        market_share_proxy=market_share_proxy,
        stability_score=stability_score,
        risk_divergence_score=risk_divergence_score,
    )

    return CompetitiveBenchmarkMetrics(
        relative_growth_index=_round_or_none(relative_growth_index),
        market_share_proxy=_round_or_none(market_share_proxy),
        stability_score=stability_score,
        momentum_classification=momentum,
        risk_divergence_score=_round_or_none(risk_divergence_score),
        explainability={
            "formulas": {
                "relative_growth_index": "(1 + entity_growth) / (1 + market_avg_growth)",
                "market_share_proxy": "entity_mrr / (entity_mrr + sum(peer_mrr))",
                "stability_score": "100 / (1 + (volatility / max(abs(mean), epsilon)))",
                "risk_divergence_score": "50 + 50 * ((market_risk - entity_risk) / abs(market_risk))",
            },
            "thresholds": dict(_MOMENTUM_THRESHOLDS),
            "inputs": {
                "growth_metric": growth_metric,
                "entity_growth": _round_or_none(client_growth),
                "market_avg_growth": _round_or_none(market_growth_avg),
                "mrr_metric": mrr_metric,
                "entity_mrr": _round_or_none(client_mrr),
                "total_market_mrr": _round_or_none(total_market_mrr),
                "stability_source": stability_source,
                "volatility_coefficient": _round_or_none(volatility_coefficient),
                "risk_metric": risk_metric,
                "entity_risk": _round_or_none(client_risk),
                "market_avg_risk": _round_or_none(market_risk_avg),
            },
        },
    )


def score_composite(
    client_metrics: dict[str, float],
    benchmark_data: dict[str, Sequence[Any] | np.ndarray],
    *,
    metric_categories: Mapping[str, Sequence[str]] | None = None,
    category_weights: Mapping[str, float] | None = None,
    metric_directions: Mapping[str, Literal["higher_is_better", "lower_is_better"]] | None = None,
    use_confidence_weighting: bool | None = None,
    client_confidences: Mapping[str, float] | None = None,
    benchmark_confidences: Mapping[str, float | Sequence[Any] | np.ndarray] | None = None,
    use_executive_formula: bool = True,
    use_macro_adjustment: bool | None = None,
    client_inflation_rates: Mapping[str, float] | None = None,
    benchmark_inflation_rates: Mapping[str, float | Sequence[Any] | np.ndarray] | None = None,
    adjustment_modes: Mapping[str, str] | None = None,
    metric_series: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    growth_metric_candidates: Sequence[str] | None = None,
    mrr_metric_candidates: Sequence[str] | None = None,
    risk_metric_candidates: Sequence[str] | None = None,
    stability_metric_candidates: Sequence[str] | None = None,
) -> CompositeScore:
    """Compute a weighted composite competitive score.

    Parameters
    ----------
    client_metrics:
        ``{metric_name: client_value}`` mapping.
    benchmark_data:
        ``{metric_name: array_of_benchmark_values}`` mapping.
    metric_categories:
        Optional override for the category → metric-names mapping.
        Defaults to ``DEFAULT_METRIC_CATEGORIES``.
    category_weights:
        Optional weight overrides. Merged with ``DEFAULT_CATEGORY_WEIGHTS``
        and re-normalised to sum to 1.0.

    Returns
    -------
    CompositeScore
        Overall weighted score, per-category scores, weakest/strongest
        metric, and full per-metric detail.
    """
    # 1. Score every metric individually
    details = score_metrics_batch(
        client_metrics,
        benchmark_data,
        metric_directions=metric_directions,
        use_confidence_weighting=use_confidence_weighting,
        client_confidences=client_confidences,
        benchmark_confidences=benchmark_confidences,
        use_macro_adjustment=use_macro_adjustment,
        client_inflation_rates=client_inflation_rates,
        benchmark_inflation_rates=benchmark_inflation_rates,
        adjustment_modes=adjustment_modes,
    )
    resolved_cfg = get_scoring_config()
    apply_confidence_weighting = (
        use_confidence_weighting
        if use_confidence_weighting is not None
        else resolved_cfg.use_confidence_weighting
    )

    if not details:
        return CompositeScore(
            overall_score=50.0,
            category_scores={},
            weakest_metric=None,
            strongest_metric=None,
            metric_details={},
            base_overall_score=50.0,
            growth_score=50.0,
            level_score=50.0,
            stability_score=50.0,
            confidence_score=50.0,
            executive_formula_applied=bool(use_executive_formula),
            confidence_weighting_applied=bool(apply_confidence_weighting),
        )

    # 2. Build category lookup
    cat_map = dict(metric_categories) if metric_categories else dict(DEFAULT_METRIC_CATEGORIES)
    cat_index = _build_category_index(cat_map)

    # 3. Group selected metric scores by category
    metric_scores: dict[str, float] = {}
    grouped: dict[str, list[float]] = {}
    for name, rs in details.items():
        metric_score = (
            rs.confidence_adjusted_score
            if apply_confidence_weighting
            else rs.normalised_score
        )
        metric_scores[name] = float(metric_score)

        category = cat_index.get(name.strip().lower())
        if category is None:
            continue
        grouped.setdefault(category, []).append(metric_score)

    # 4. Per-category score = mean of member normalised scores
    category_scores: dict[str, float] = {}
    for cat, scores in grouped.items():
        category_scores[cat] = round(float(np.mean(scores)), 2)

    # 5. Weighted overall score
    weights = _resolve_category_weights(category_weights)
    weighted_sum = 0.0
    weight_total = 0.0
    for cat, cat_score in category_scores.items():
        w = weights.get(cat, 0.0)
        weighted_sum += cat_score * w
        weight_total += w

    if weight_total > 0.0:
        legacy_overall = round(weighted_sum / weight_total, 2)
    else:
        # Fallback: simple mean of all category scores
        legacy_overall = round(float(np.mean(list(category_scores.values()))), 2) if category_scores else 50.0

    legacy_overall = float(np.clip(legacy_overall, 0.0, 100.0))

    growth_score = category_scores.get("growth")
    if growth_score is None:
        growth_score = round(float(np.mean(list(metric_scores.values()))), 2) if metric_scores else 50.0
    growth_score = float(np.clip(growth_score, 0.0, 100.0))

    level_categories = [cat for cat in category_scores if cat != "growth"]
    if level_categories:
        level_weight_total = sum(weights.get(cat, 0.0) for cat in level_categories)
        if level_weight_total > 0.0:
            level_score = sum(
                category_scores[cat] * (weights.get(cat, 0.0) / level_weight_total)
                for cat in level_categories
            )
        else:
            level_score = float(np.mean([category_scores[cat] for cat in level_categories]))
    else:
        level_score = float(np.mean(list(metric_scores.values()))) if metric_scores else 50.0
    level_score = float(np.clip(round(level_score, 2), 0.0, 100.0))

    macro_applied_values = [
        rs.macro_resilience
        for rs in details.values()
        if rs.macro_adjustment_applied
    ]
    if macro_applied_values:
        stability_score = float(np.mean(macro_applied_values))
    else:
        values = list(metric_scores.values())
        if len(values) <= 1:
            stability_score = 100.0
        else:
            dispersion = float(np.std(np.asarray(values, dtype=float), ddof=0))
            stability_score = 100.0 - dispersion
    stability_score = float(np.clip(round(stability_score, 2), 0.0, 100.0))

    confidence_values = [float(rs.effective_confidence) * 100.0 for rs in details.values()]
    confidence_score = (
        float(np.mean(confidence_values))
        if confidence_values
        else 100.0
    )
    confidence_score = float(np.clip(round(confidence_score, 2), 0.0, 100.0))

    competitive_metrics = _compute_competitive_benchmark_metrics(
        client_metrics=client_metrics,
        benchmark_data=benchmark_data,
        metric_scores=metric_scores,
        metric_series=metric_series,
        growth_metric_candidates=(
            tuple(growth_metric_candidates)
            if growth_metric_candidates is not None
            else DEFAULT_GROWTH_METRIC_CANDIDATES
        ),
        mrr_metric_candidates=(
            tuple(mrr_metric_candidates)
            if mrr_metric_candidates is not None
            else DEFAULT_MRR_METRIC_CANDIDATES
        ),
        risk_metric_candidates=(
            tuple(risk_metric_candidates)
            if risk_metric_candidates is not None
            else DEFAULT_RISK_METRIC_CANDIDATES
        ),
        stability_metric_candidates=(
            tuple(stability_metric_candidates)
            if stability_metric_candidates is not None
            else DEFAULT_STABILITY_SERIES_CANDIDATES
        ),
        zero_guard=resolved_cfg.zero_guard,
    )

    executive_overall = (
        growth_score * 0.4
        + level_score * 0.3
        + stability_score * 0.2
        + confidence_score * 0.1
    )
    executive_overall = float(np.clip(round(executive_overall, 2), 0.0, 100.0))
    overall = executive_overall if use_executive_formula else legacy_overall

    # 6. Weakest / strongest metric
    sorted_metrics = sorted(
        details.items(),
        key=lambda kv: (
            kv[1].confidence_adjusted_score
            if apply_confidence_weighting
            else kv[1].normalised_score
        ),
    )
    weakest = sorted_metrics[0][0]
    strongest = sorted_metrics[-1][0]

    return CompositeScore(
        overall_score=overall,
        category_scores=category_scores,
        weakest_metric=weakest,
        strongest_metric=strongest,
        metric_details=details,
        base_overall_score=legacy_overall,
        growth_score=growth_score,
        level_score=level_score,
        stability_score=stability_score,
        confidence_score=confidence_score,
        executive_formula_applied=bool(use_executive_formula),
        confidence_weighting_applied=bool(apply_confidence_weighting),
        competitive_metrics=competitive_metrics,
    )
