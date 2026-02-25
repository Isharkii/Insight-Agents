from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

DecayMode = Literal["linear", "exponential"]
SeverityLevel = Literal["mild", "moderate", "severe"]

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


_EROSION_RULES = _as_dict(_load_business_rules().get("competitive_erosion"))
_EROSION_DEFAULTS = _as_dict(_EROSION_RULES.get("defaults"))
_EROSION_SEVERITY = _as_dict(_EROSION_RULES.get("severity"))


@dataclass(frozen=True)
class CompetitiveErosionConfig:
    horizon_periods: int = _as_int(_EROSION_DEFAULTS.get("horizon_periods"), 8)
    decay_mode: str = str(_EROSION_DEFAULTS.get("decay_mode", "linear")).strip().lower() or "linear"
    min_erosion: float = _as_float(_EROSION_DEFAULTS.get("min_erosion"), 0.0)
    max_erosion: float = _as_float(_EROSION_DEFAULTS.get("max_erosion"), 0.95)
    max_cumulative_erosion: float = _as_float(_EROSION_DEFAULTS.get("max_cumulative_erosion"), 0.99)
    noise_std: float = _as_float(_EROSION_DEFAULTS.get("noise_std"), 0.0)
    zero_guard: float = _as_float(_EROSION_DEFAULTS.get("zero_guard"), 1e-9)

    def severity_multiplier(self, severity: SeverityLevel | str) -> float:
        key = str(severity or "moderate").strip().lower() or "moderate"
        payload = _as_dict(_EROSION_SEVERITY.get(key))
        if payload:
            return max(0.0, _as_float(payload.get("multiplier"), 1.0))
        if key == "mild":
            return 0.75
        if key == "severe":
            return 1.35
        return 1.0


def score_to_vulnerability(competitive_score: Any) -> float:
    """
    Convert a 0-100 competitive score to vulnerability in [0, 1].

    Lower score means weaker positioning and higher vulnerability.
    """

    score = _coerce_float(competitive_score)
    if score is None:
        score = 50.0
    bounded = min(100.0, max(0.0, float(score)))
    return round((100.0 - bounded) / 100.0, 6)


def compute_erosion_curve(
    horizon_periods: int,
    *,
    effective_erosion: float,
    decay_mode: DecayMode | str,
    max_cumulative_erosion: float = 0.99,
    seed: int | None = None,
    noise_std: float = 0.0,
) -> np.ndarray:
    """
    Build cumulative erosion intensity per period.

    - linear:    erosion_t = effective_erosion * t
    - exponential: erosion_t = 1 - exp(-effective_erosion * t)
    """

    horizon = max(0, int(horizon_periods))
    if horizon == 0:
        return np.array([], dtype=np.float64)

    effective = max(0.0, float(effective_erosion))
    t = np.arange(1, horizon + 1, dtype=np.float64)
    mode = str(decay_mode or "linear").strip().lower()

    if mode == "exponential":
        cumulative = 1.0 - np.exp(-effective * t)
    else:
        cumulative = effective * t

    cap = min(0.999999, max(0.0, float(max_cumulative_erosion)))
    cumulative = np.clip(cumulative, 0.0, cap)

    # Random perturbation is optional and deterministic only when a seed is provided.
    if seed is not None and float(noise_std) > 0.0:
        rng = np.random.default_rng(int(seed))
        noise = rng.normal(0.0, float(noise_std), size=horizon)
        cumulative = np.clip(cumulative + noise, 0.0, cap)
        cumulative = np.maximum.accumulate(cumulative)

    return cumulative.astype(np.float64, copy=False)


def apply_erosion_to_growth(
    industry_growth_rate: Any,
    erosion_curve: np.ndarray,
    *,
    min_growth_floor: float = -0.95,
) -> np.ndarray:
    """
    Apply cumulative erosion to a baseline growth rate.

    Economic interpretation:
    - Positive baseline growth declines toward zero as competitive pressure rises.
    - Negative baseline growth becomes more negative under additional erosion.
    """

    base = _coerce_float(industry_growth_rate)
    if base is None:
        base = 0.0

    growth = np.full(erosion_curve.shape, float(base), dtype=np.float64)
    erosion = np.clip(erosion_curve.astype(np.float64, copy=False), 0.0, 0.999999)
    adjusted = np.where(growth >= 0.0, growth * (1.0 - erosion), growth * (1.0 + erosion))
    return np.maximum(adjusted, float(min_growth_floor))


def project_market_share(
    current_market_share: Any,
    erosion_curve: np.ndarray,
) -> np.ndarray:
    """
    Project market share using incremental erosion implied by cumulative curve.

    Vectorized form:
    share_t = share_0 * prod(1 - incremental_erosion_i), i=1..t
    """

    share_fraction, input_scale = _normalize_market_share(current_market_share)
    if erosion_curve.size == 0:
        return np.array([], dtype=np.float64)

    cumulative = np.clip(erosion_curve.astype(np.float64, copy=False), 0.0, 0.999999)
    incremental = np.diff(np.concatenate(([0.0], cumulative)))
    incremental = np.clip(incremental, 0.0, 0.999999)
    out = float(share_fraction) * np.cumprod(1.0 - incremental, dtype=np.float64)
    out = np.clip(out, 0.0, 1.0)

    if input_scale == "percent":
        return out * 100.0
    return out


def compute_erosion_impact_pct(
    *,
    baseline_growth_series: np.ndarray,
    adjusted_growth_series: np.ndarray,
    starting_market_share: float,
    ending_market_share: float,
    zero_guard: float = 1e-9,
) -> float:
    """
    Compute a blended impact percent from growth erosion and share loss.
    """

    guard = max(1e-12, float(zero_guard))
    base_mean = float(np.nanmean(baseline_growth_series)) if baseline_growth_series.size else 0.0
    adj_mean = float(np.nanmean(adjusted_growth_series)) if adjusted_growth_series.size else 0.0
    growth_impact = (base_mean - adj_mean) / max(abs(base_mean), guard)
    share_impact = (starting_market_share - ending_market_share) / max(abs(starting_market_share), guard)

    blended = max(0.0, (0.5 * growth_impact) + (0.5 * share_impact))
    return round(blended * 100.0, 6)


def simulate_competitive_erosion(
    *,
    current_market_share: Any,
    competitive_score: Any,
    industry_growth_rate: Any,
    erosion_factor: Any,
    horizon_periods: int | None = None,
    severity: SeverityLevel | str = "moderate",
    decay_mode: DecayMode | str | None = None,
    seed: int | None = None,
    noise_std: float | None = None,
    config: CompetitiveErosionConfig | None = None,
) -> dict[str, Any]:
    """
    Simulate deterministic competitive erosion trajectories.

    Inputs:
    - current market share
    - competitive score (0-100)
    - industry growth rate
    - erosion factor
    - severity: mild/moderate/severe
    - decay mode: linear/exponential

    Returns:
    {
      "adjusted_growth_rate_series": [...],
      "market_share_series": [...],
      "erosion_impact_pct": <float>
    }
    """

    cfg = config or CompetitiveErosionConfig()
    horizon = int(horizon_periods) if horizon_periods is not None else int(cfg.horizon_periods)
    horizon = max(0, horizon)

    vulnerability = score_to_vulnerability(competitive_score)
    severity_multiplier = cfg.severity_multiplier(severity)
    raw_erosion = _coerce_float(erosion_factor)
    if raw_erosion is None:
        raw_erosion = 0.0

    min_erosion = max(0.0, float(cfg.min_erosion))
    max_erosion = max(min_erosion, float(cfg.max_erosion))
    effective_erosion = float(raw_erosion) * vulnerability * severity_multiplier
    effective_erosion = max(min_erosion, min(max_erosion, effective_erosion))

    resolved_decay_mode = (str(decay_mode).strip().lower() if decay_mode is not None else str(cfg.decay_mode).strip().lower())
    if resolved_decay_mode not in {"linear", "exponential"}:
        resolved_decay_mode = "linear"

    resolved_noise_std = float(cfg.noise_std if noise_std is None else noise_std)
    erosion_curve = compute_erosion_curve(
        horizon,
        effective_erosion=effective_erosion,
        decay_mode=resolved_decay_mode,
        max_cumulative_erosion=cfg.max_cumulative_erosion,
        seed=seed,
        noise_std=resolved_noise_std,
    )

    adjusted_growth = apply_erosion_to_growth(industry_growth_rate, erosion_curve)
    market_share = project_market_share(current_market_share, erosion_curve)

    baseline_growth = np.full(adjusted_growth.shape, _coerce_float(industry_growth_rate) or 0.0, dtype=np.float64)
    start_share, start_scale = _normalize_market_share(current_market_share)
    if market_share.size > 0:
        final_share_raw = float(market_share[-1])
        final_share = final_share_raw / 100.0 if start_scale == "percent" else final_share_raw
    else:
        final_share = start_share

    impact_pct = compute_erosion_impact_pct(
        baseline_growth_series=baseline_growth,
        adjusted_growth_series=adjusted_growth,
        starting_market_share=start_share,
        ending_market_share=final_share,
        zero_guard=cfg.zero_guard,
    )

    return {
        "adjusted_growth_rate_series": _to_optional_list(adjusted_growth),
        "market_share_series": _to_optional_list(market_share),
        "erosion_impact_pct": impact_pct,
    }


def _normalize_market_share(value: Any) -> tuple[float, str]:
    parsed = _coerce_float(value)
    if parsed is None:
        return 0.0, "fraction"

    share = max(0.0, float(parsed))
    if share > 1.0 and share <= 100.0:
        return min(share / 100.0, 1.0), "percent"
    return min(share, 1.0), "fraction"


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _to_optional_list(values: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for value in values.astype(np.float64, copy=False):
        if np.isfinite(value):
            out.append(round(float(value), 6))
        else:
            out.append(None)
    return out
