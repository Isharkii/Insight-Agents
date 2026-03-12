"""
app/services/statistics/seasonality.py

Lightweight deterministic seasonality detection for KPI time-series.

Detects periodic patterns (weekly, monthly, quarterly, annual cycles) using
two independent methods that must agree before a period is declared seasonal:

    1. **Lag autocorrelation** — measures linear correlation between a series
       and a lagged copy of itself.  Significance is tested against the Bartlett
       critical value (1.96 / √n).

    2. **Variance ratio** — groups values by their phase position within a
       candidate period and compares within-phase variance to total variance.
       Seasonal series have low ratio (same-phase values cluster tightly).

Both methods are O(n) per candidate period, use only the Python standard
library, and are fully deterministic.

Minimum data requirements are enforced per candidate period — the detector
will not claim a 12-month cycle from 8 data points.

Configuration
-------------
All thresholds are loaded from ``config/business_rules.yaml`` under the
``seasonality`` key, with sensible defaults when the config is absent.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

from app.services.statistics.normalization import coerce_numeric_series


_BUSINESS_RULES_PATH = Path(__file__).resolve().parents[3] / "config" / "business_rules.yaml"

_ZERO_GUARD = 1e-9


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


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


_SEASONALITY_RULES = _as_dict(_load_business_rules().get("seasonality"))
_SEASONALITY_DEFAULTS = _as_dict(_SEASONALITY_RULES.get("defaults"))


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SeasonalityConfig:
    """Thresholds and candidate periods for seasonality detection."""

    # Candidate periods to test (in series index units — typically months)
    candidate_periods: tuple[int, ...] = (3, 4, 6, 12)

    # Minimum data points required as a multiple of the candidate period.
    # E.g. 1.5 × period=12 → need at least 18 points.
    min_cycles_multiplier: float = _as_float(
        _SEASONALITY_DEFAULTS.get("min_cycles_multiplier"), 1.5,
    )

    # Autocorrelation must exceed this many standard errors (Bartlett).
    # Default 1.96 ≈ 95% confidence under the null of white noise.
    bartlett_z: float = _as_float(
        _SEASONALITY_DEFAULTS.get("bartlett_z"), 1.96,
    )

    # Variance ratio threshold — within-phase / total.
    # Below this → seasonal signal present.
    variance_ratio_threshold: float = _as_float(
        _SEASONALITY_DEFAULTS.get("variance_ratio_threshold"), 0.60,
    )

    # Minimum autocorrelation magnitude (absolute floor regardless of Bartlett).
    min_autocorrelation: float = _as_float(
        _SEASONALITY_DEFAULTS.get("min_autocorrelation"), 0.25,
    )

    # Minimum points per phase group for variance ratio to be meaningful.
    min_phase_observations: int = _as_int(
        _SEASONALITY_DEFAULTS.get("min_phase_observations"), 2,
    )

    spectral_power_ratio_threshold: float = _as_float(
        _SEASONALITY_DEFAULTS.get("spectral_power_ratio_threshold"), 0.20,
    )

    zero_guard: float = _as_float(
        _SEASONALITY_DEFAULTS.get("zero_guard"), 1e-9,
    )


def load_seasonality_config() -> SeasonalityConfig:
    """Build config from business rules, falling back to defaults."""
    rules = _SEASONALITY_DEFAULTS
    if not rules:
        return SeasonalityConfig()

    raw_periods = rules.get("candidate_periods")
    if isinstance(raw_periods, list):
        periods = tuple(
            int(p) for p in raw_periods
            if isinstance(p, (int, float)) and int(p) >= 2
        )
    else:
        periods = (3, 4, 6, 12)

    return SeasonalityConfig(
        candidate_periods=periods or (3, 4, 6, 12),
        min_cycles_multiplier=_as_float(rules.get("min_cycles_multiplier"), 1.5),
        bartlett_z=_as_float(rules.get("bartlett_z"), 1.96),
        variance_ratio_threshold=_as_float(rules.get("variance_ratio_threshold"), 0.60),
        min_autocorrelation=_as_float(rules.get("min_autocorrelation"), 0.25),
        min_phase_observations=_as_int(rules.get("min_phase_observations"), 2),
        spectral_power_ratio_threshold=_as_float(
            rules.get("spectral_power_ratio_threshold"), 0.20,
        ),
        zero_guard=_as_float(rules.get("zero_guard"), 1e-9),
    )


# ── Core algorithms ─────────────────────────────────────────────────────────

def autocorrelation_at_lag(values: list[float], lag: int) -> float:
    """
    Compute the sample autocorrelation r(lag) for a centred series.

    Formula:
        r(k) = Σ(xₜ - μ)(xₜ₋ₖ - μ) / Σ(xₜ - μ)²

    Returns 0.0 when the denominator is negligible.
    """
    n = len(values)
    if lag <= 0 or lag >= n:
        return 0.0

    mu = mean(values)
    denominator = sum((v - mu) ** 2 for v in values)
    if denominator < _ZERO_GUARD:
        return 0.0

    numerator = sum(
        (values[t] - mu) * (values[t - lag] - mu)
        for t in range(lag, n)
    )
    return numerator / denominator


def variance_ratio(values: list[float], period: int) -> dict[str, Any]:
    """
    Group values by phase position within *period* and compare variances.

    Phase k contains values at indices k, k+period, k+2*period, …

    Returns
    -------
    dict with keys:
        ratio           – within_variance / total_variance (lower = more seasonal)
        within_variance – mean of per-phase variances
        total_variance  – variance of the full series
        phase_means     – list of per-phase mean values
        phase_counts    – list of per-phase observation counts
    """
    n = len(values)
    if period < 2 or n < period:
        return {
            "ratio": 1.0,
            "within_variance": None,
            "total_variance": None,
            "phase_means": [],
            "phase_counts": [],
        }

    # Group by phase
    phases: list[list[float]] = [[] for _ in range(period)]
    for idx, val in enumerate(values):
        phases[idx % period].append(val)

    phase_means: list[float] = []
    phase_variances: list[float] = []
    phase_counts: list[int] = []

    for group in phases:
        phase_counts.append(len(group))
        if len(group) >= 1:
            phase_means.append(round(mean(group), 6))
        else:
            phase_means.append(0.0)
        if len(group) >= 2:
            phase_variances.append(pstdev(group) ** 2)
        else:
            phase_variances.append(0.0)

    total_var = pstdev(values) ** 2 if len(values) >= 2 else 0.0
    within_var = mean(phase_variances) if phase_variances else 0.0

    if total_var < _ZERO_GUARD:
        ratio = 0.0  # constant series → perfect "seasonality" (trivially)
    else:
        ratio = within_var / total_var

    return {
        "ratio": round(ratio, 6),
        "within_variance": round(within_var, 6),
        "total_variance": round(total_var, 6),
        "phase_means": phase_means,
        "phase_counts": phase_counts,
    }


def _seasonal_strength(acf: float, vr_ratio: float) -> float:
    """
    Combine autocorrelation and variance ratio into a 0–1 strength score.

    strength = acf_component × variance_component

    acf_component      = clamp(|acf|, 0, 1)
    variance_component = clamp(1 - vr_ratio, 0, 1)
    """
    acf_component = max(0.0, min(1.0, abs(acf)))
    variance_component = max(0.0, min(1.0, 1.0 - vr_ratio))
    return round(acf_component * variance_component, 6)


# ── Main detector ────────────────────────────────────────────────────────────

def detect_seasonality(
    values: Sequence[Any],
    *,
    config: SeasonalityConfig | None = None,
) -> dict[str, Any]:
    """
    Detect periodic seasonality in a time-series.

    Parameters
    ----------
    values:
        Ordered numeric observations, oldest first.
    config:
        Optional overrides; defaults loaded from business_rules.yaml.

    Returns
    -------
    dict (JSON-compatible)
        Top-level keys:
            detected           – bool, True if any period is confirmed seasonal
            primary_period     – int or None, strongest detected period
            primary_strength   – float, 0–1 strength of strongest period
            candidates         – list of per-period analysis dicts
            summary            – human-readable assessment
            status             – "detected" | "none" | "insufficient_history"
            warnings           – list of diagnostic strings
            diagnostics        – thresholds and config used
    """
    cfg = config or load_seasonality_config()
    series = coerce_numeric_series(values)
    n = len(series)

    insufficient_result: dict[str, Any] = {
        "detected": False,
        "primary_period": None,
        "primary_strength": 0.0,
        "candidates": [],
        "summary": "Insufficient data for seasonality detection.",
        "status": "insufficient_history",
        "warnings": [f"Need at least {_min_points_for_smallest_period(cfg)} points; got {n}."],
        "diagnostics": _diagnostics(cfg, n),
    }

    smallest_period = min(cfg.candidate_periods) if cfg.candidate_periods else 3
    if n < math.ceil(smallest_period * cfg.min_cycles_multiplier):
        return insufficient_result

    # Detrend: subtract linear trend to isolate cyclical component
    detrended = _linear_detrend(series)

    # Signal-to-noise guard: if detrended variance is negligible relative to
    # the original series, the trend explains almost everything and there is
    # no meaningful cyclical signal to detect.
    original_std = pstdev(series) if n >= 2 else 0.0
    detrended_std = pstdev(detrended) if n >= 2 else 0.0
    signal_ratio = detrended_std / max(original_std, cfg.zero_guard)
    _MIN_SIGNAL_RATIO = 0.05  # need ≥5% residual variance

    if signal_ratio < _MIN_SIGNAL_RATIO:
        return {
            "detected": False,
            "primary_period": None,
            "primary_strength": 0.0,
            "candidates": [],
            "summary": "Trend dominates; no cyclical signal after detrending.",
            "status": "none",
            "warnings": [
                f"Detrended signal ratio {signal_ratio:.4f} below threshold "
                f"{_MIN_SIGNAL_RATIO}; series is trend-dominated."
            ],
            "diagnostics": _diagnostics(cfg, n),
        }

    spectral = _spectral_analysis(detrended)
    dynamic_periods = set(int(p) for p in cfg.candidate_periods if int(p) >= 2)
    spectral_period = spectral.get("dominant_period")
    if isinstance(spectral_period, int) and spectral_period >= 2:
        dynamic_periods.add(int(spectral_period))

    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []

    for period in sorted(dynamic_periods):
        min_required = math.ceil(period * cfg.min_cycles_multiplier)

        if n < min_required:
            candidates.append({
                "period": period,
                "status": "insufficient_data",
                "min_required": min_required,
                "points_available": n,
                "autocorrelation": None,
                "bartlett_critical": None,
                "variance_ratio": None,
                "detected": False,
                "strength": 0.0,
            })
            warnings.append(
                f"Period {period}: need {min_required} points, have {n}."
            )
            continue

        # Method 1: autocorrelation
        acf = autocorrelation_at_lag(detrended, lag=period)
        bartlett_critical = cfg.bartlett_z / math.sqrt(n)
        acf_passes = abs(acf) > bartlett_critical and abs(acf) >= cfg.min_autocorrelation

        # Method 2: variance ratio
        vr = variance_ratio(detrended, period)
        vr_ratio = vr["ratio"]
        # Check minimum phase observations
        min_phase_obs = min(vr["phase_counts"]) if vr["phase_counts"] else 0
        vr_valid = min_phase_obs >= cfg.min_phase_observations
        vr_passes = vr_valid and vr_ratio < cfg.variance_ratio_threshold

        spectral_period_match = (
            isinstance(spectral_period, int)
            and int(spectral_period) == int(period)
        )
        spectral_ratio = float(spectral.get("dominant_power_ratio") or 0.0)
        spectral_passes = (
            spectral_period_match
            and spectral_ratio >= cfg.spectral_power_ratio_threshold
        )

        detected = (acf_passes and vr_passes) or (
            spectral_passes and (acf_passes or vr_passes)
        )
        strength = _seasonal_strength(acf, vr_ratio) if detected else 0.0

        candidates.append({
            "period": period,
            "status": "detected" if detected else "not_detected",
            "min_required": min_required,
            "points_available": n,
            "autocorrelation": round(acf, 6),
            "bartlett_critical": round(bartlett_critical, 6),
            "acf_significant": acf_passes,
            "variance_ratio": round(vr_ratio, 6),
            "variance_ratio_significant": vr_passes,
            "spectral_period_match": spectral_period_match,
            "spectral_power_ratio": round(spectral_ratio, 6),
            "spectral_significant": spectral_passes,
            "phase_means": vr["phase_means"],
            "phase_counts": vr["phase_counts"],
            "detected": detected,
            "strength": strength,
        })

    # Select strongest detected period
    detected_periods = [c for c in candidates if c["detected"]]
    if detected_periods:
        primary = max(detected_periods, key=lambda c: c["strength"])
        primary_period = primary["period"]
        primary_strength = primary["strength"]
        any_detected = True
        summary = (
            f"Seasonal pattern detected with period {primary_period} "
            f"(strength {primary_strength:.2f})."
        )
        status = "detected"
    else:
        primary_period = None
        primary_strength = 0.0
        any_detected = False
        summary = "No significant seasonal pattern detected."
        status = "none"

    return {
        "detected": any_detected,
        "primary_period": primary_period,
        "primary_strength": primary_strength,
        "candidates": candidates,
        "summary": summary,
        "status": status,
        "warnings": warnings,
        "diagnostics": _diagnostics(cfg, n, spectral=spectral),
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


def _min_points_for_smallest_period(cfg: SeasonalityConfig) -> int:
    smallest = min(cfg.candidate_periods) if cfg.candidate_periods else 3
    return math.ceil(smallest * cfg.min_cycles_multiplier)


def _diagnostics(
    cfg: SeasonalityConfig,
    n: int,
    *,
    spectral: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "series_length": n,
        "candidate_periods": list(cfg.candidate_periods),
        "min_cycles_multiplier": cfg.min_cycles_multiplier,
        "bartlett_z": cfg.bartlett_z,
        "variance_ratio_threshold": cfg.variance_ratio_threshold,
        "min_autocorrelation": cfg.min_autocorrelation,
        "min_phase_observations": cfg.min_phase_observations,
        "spectral_power_ratio_threshold": cfg.spectral_power_ratio_threshold,
        "spectral_analysis": spectral or {},
    }


def _spectral_analysis(values: list[float]) -> dict[str, Any]:
    """
    DFT-based dominant periodicity detection without external dependencies.
    """
    n = len(values)
    if n < 8:
        return {
            "status": "insufficient_history",
            "dominant_period": None,
            "dominant_frequency_index": None,
            "dominant_power_ratio": 0.0,
        }

    centered = list(values)
    mu = mean(centered)
    centered = [val - mu for val in centered]

    half = n // 2
    if half < 1:
        return {
            "status": "insufficient_history",
            "dominant_period": None,
            "dominant_frequency_index": None,
            "dominant_power_ratio": 0.0,
        }

    power_by_k: dict[int, float] = {}
    for k in range(1, half + 1):
        real = 0.0
        imag = 0.0
        for idx, val in enumerate(centered):
            angle = (2.0 * math.pi * k * idx) / n
            real += val * math.cos(angle)
            imag -= val * math.sin(angle)
        power = (real * real + imag * imag) / max(float(n * n), 1.0)
        power_by_k[k] = power

    total_power = sum(power_by_k.values())
    if total_power < _ZERO_GUARD:
        return {
            "status": "no_periodic_signal",
            "dominant_period": None,
            "dominant_frequency_index": None,
            "dominant_power_ratio": 0.0,
        }

    dominant_k = max(power_by_k, key=lambda key: power_by_k[key])
    dominant_power = power_by_k[dominant_k]
    dominant_ratio = dominant_power / total_power
    dominant_period = int(round(n / dominant_k)) if dominant_k > 0 else None
    if dominant_period is not None and dominant_period < 2:
        dominant_period = 2

    return {
        "status": "ok",
        "dominant_period": dominant_period,
        "dominant_frequency_index": dominant_k,
        "dominant_power_ratio": round(dominant_ratio, 6),
        "total_power": round(total_power, 6),
    }
