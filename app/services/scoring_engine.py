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
from typing import Any, Mapping, Sequence

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


def _safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size > 0 else 0.0


def _safe_median(arr: np.ndarray) -> float:
    return float(np.median(arr)) if arr.size > 0 else 0.0


def _safe_std(arr: np.ndarray) -> float:
    """Population standard deviation (ddof=0)."""
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=0))


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
) -> dict[str, float | str]:
    bench = _coerce_benchmark(benchmark_values)
    b_mean = _safe_mean(bench)
    b_median = _safe_median(bench)
    b_std = _safe_std(bench)

    pct = percentile_rank(client_value, bench)

    if bench.size == 0 or (bench.size < 2 and b_std == 0.0):
        z = 0.0
        dev = 0.0
    else:
        z = zscore(client_value, b_mean, b_std)
        dev = deviation_pct(client_value, b_median)

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
    nominal = _score_metric_core(metric_name, client_value, benchmark_values)
    nominal_score = float(nominal["normalised_score"])
    resolved_cfg = get_scoring_config()
    apply_macro = (
        use_macro_adjustment
        if use_macro_adjustment is not None
        else resolved_cfg.use_macro_adjustment
    )

    client_rate = _normalize_rate(float(client_inflation_rate)) if client_inflation_rate is not None else float("nan")
    bench_rate_source = benchmark_inflation_rate if benchmark_inflation_rate is not None else client_inflation_rate

    if not apply_macro:
        return RelativeScore(
            **nominal,
            nominal_score=nominal_score,
            real_score=nominal_score,
            macro_resilience=100.0,
            delta_due_to_macro=0.0,
            macro_adjustment_applied=False,
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

    real = _score_metric_core(metric_name, real_client, real_bench_arr)
    real_score = float(real["normalised_score"])
    delta = round(real_score - nominal_score, 4)
    resilience = _compute_macro_resilience(
        nominal_score=nominal_score,
        real_score=real_score,
        sensitivity=resolved_cfg.macro_resilience_sensitivity,
    )

    return RelativeScore(
        **real,
        nominal_score=round(nominal_score, 4),
        real_score=round(real_score, 4),
        macro_resilience=resilience,
        delta_due_to_macro=delta,
        macro_adjustment_applied=True,
    )


def score_metrics_batch(
    client_metrics: dict[str, float],
    benchmark_data: dict[str, Sequence[Any] | np.ndarray],
    *,
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


def score_composite(
    client_metrics: dict[str, float],
    benchmark_data: dict[str, Sequence[Any] | np.ndarray],
    *,
    metric_categories: Mapping[str, Sequence[str]] | None = None,
    category_weights: Mapping[str, float] | None = None,
    use_macro_adjustment: bool | None = None,
    client_inflation_rates: Mapping[str, float] | None = None,
    benchmark_inflation_rates: Mapping[str, float | Sequence[Any] | np.ndarray] | None = None,
    adjustment_modes: Mapping[str, str] | None = None,
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
        use_macro_adjustment=use_macro_adjustment,
        client_inflation_rates=client_inflation_rates,
        benchmark_inflation_rates=benchmark_inflation_rates,
        adjustment_modes=adjustment_modes,
    )

    if not details:
        return CompositeScore(
            overall_score=50.0,
            category_scores={},
            weakest_metric=None,
            strongest_metric=None,
            metric_details={},
        )

    # 2. Build category lookup
    cat_map = dict(metric_categories) if metric_categories else dict(DEFAULT_METRIC_CATEGORIES)
    cat_index = _build_category_index(cat_map)

    # 3. Group normalised scores by category
    grouped: dict[str, list[float]] = {}
    for name, rs in details.items():
        category = cat_index.get(name.strip().lower())
        if category is None:
            continue
        grouped.setdefault(category, []).append(rs.normalised_score)

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
        overall = round(weighted_sum / weight_total, 2)
    else:
        # Fallback: simple mean of all category scores
        overall = round(float(np.mean(list(category_scores.values()))), 2) if category_scores else 50.0

    overall = float(np.clip(overall, 0.0, 100.0))

    # 6. Weakest / strongest metric
    sorted_metrics = sorted(details.items(), key=lambda kv: kv[1].normalised_score)
    weakest = sorted_metrics[0][0]
    strongest = sorted_metrics[-1][0]

    return CompositeScore(
        overall_score=overall,
        category_scores=category_scores,
        weakest_metric=weakest,
        strongest_metric=strongest,
        metric_details=details,
    )
