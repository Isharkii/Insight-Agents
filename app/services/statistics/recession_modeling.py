from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

RecoveryCurve = Literal["v_shape", "u_shape", "l_shape"]

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


def _as_int(value: object, default: int, *, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


_RECESSION_RULES = _as_dict(_load_business_rules().get("recession_modeling"))
_RECESSION_DEFAULTS = _as_dict(_RECESSION_RULES.get("defaults"))
_RECESSION_CURVE_PROFILES = _as_dict(_RECESSION_RULES.get("curve_profiles"))


@dataclass(frozen=True)
class RecessionCurveProfile:
    recovery_quarters: int
    plateau_quarters: int
    target_ratio: float


@dataclass(frozen=True)
class RecessionModelConfig:
    default_recovery_curve: str = str(_RECESSION_DEFAULTS.get("recovery_curve", "u_shape")).strip().lower() or "u_shape"
    gdp_weight: float = _as_float(_RECESSION_DEFAULTS.get("gdp_weight"), 0.65)
    interest_weight: float = _as_float(_RECESSION_DEFAULTS.get("interest_weight"), 0.35)
    min_shock_multiplier: float = _as_float(_RECESSION_DEFAULTS.get("min_shock_multiplier"), 0.35)
    max_shock_multiplier: float = _as_float(_RECESSION_DEFAULTS.get("max_shock_multiplier"), 1.0)
    recovery_threshold: float = _as_float(_RECESSION_DEFAULTS.get("recovery_threshold"), 0.98)
    percent_threshold: float = _as_float(_RECESSION_DEFAULTS.get("percent_threshold"), 1.0)
    percent_ceiling: float = _as_float(_RECESSION_DEFAULTS.get("percent_ceiling"), 500.0)
    zero_guard: float = _as_float(_RECESSION_DEFAULTS.get("zero_guard"), 1e-9)

    def curve_profile(self, curve: str) -> RecessionCurveProfile:
        key = str(curve or "").strip().lower()
        payload = _as_dict(_RECESSION_CURVE_PROFILES.get(key))
        if payload:
            return RecessionCurveProfile(
                recovery_quarters=_as_int(payload.get("recovery_quarters"), 4, minimum=1),
                plateau_quarters=_as_int(payload.get("plateau_quarters"), 0, minimum=0),
                target_ratio=max(0.0, _as_float(payload.get("target_ratio"), 1.0)),
            )

        if key == "v_shape":
            return RecessionCurveProfile(recovery_quarters=2, plateau_quarters=0, target_ratio=1.0)
        if key == "l_shape":
            return RecessionCurveProfile(recovery_quarters=8, plateau_quarters=2, target_ratio=0.85)
        return RecessionCurveProfile(recovery_quarters=4, plateau_quarters=1, target_ratio=1.0)


def extract_macro_shock_inputs(
    macro_metric_rows: Sequence[Mapping[str, Any]],
    *,
    country_code: str | None = None,
    percent_threshold: float = 1.0,
    percent_ceiling: float = 500.0,
) -> dict[str, Any]:
    """
    Derive GDP contraction and policy-rate spike from macro_metrics schema rows.

    Expected row compatibility:
    - metric_name, country_code, period_start, period_end, value
    """

    rows = _filter_macro_rows(macro_metric_rows, country_code=country_code)
    gdp_rows = [row for row in rows if _is_gdp_metric(str(row.get("metric_name") or ""))]
    rate_rows = [row for row in rows if _is_policy_rate_metric(str(row.get("metric_name") or ""))]

    gdp_series = _series_from_rows(gdp_rows, percent_threshold=percent_threshold, percent_ceiling=percent_ceiling)
    rate_series = _series_from_rows(rate_rows, percent_threshold=percent_threshold, percent_ceiling=percent_ceiling)

    gdp_contraction = _derive_gdp_contraction(gdp_series)
    rate_spike = _derive_interest_rate_spike(rate_series)

    return {
        "gdp_contraction_rate": _round_or_none(gdp_contraction),
        "interest_rate_spike": _round_or_none(rate_spike),
        "source_rows": {
            "gdp": len(gdp_series),
            "policy_rate": len(rate_series),
        },
    }


def compute_revenue_shock_multiplier(
    *,
    gdp_contraction_rate: Any,
    interest_rate_spike: Any,
    industry_sensitivity_coefficient: Any,
    gdp_weight: float,
    interest_weight: float,
    min_multiplier: float,
    max_multiplier: float,
) -> float:
    """
    Compute trough revenue multiplier from macro shock intensity.

    Formula:
    multiplier = 1 - sensitivity * (gdp_weight * gdp_contraction + interest_weight * interest_spike)
    """

    gdp = max(0.0, _normalize_rate(gdp_contraction_rate))
    rate = max(0.0, _normalize_rate(interest_rate_spike))
    sensitivity = max(0.0, _coerce_float(industry_sensitivity_coefficient) or 1.0)

    intensity = sensitivity * ((max(0.0, gdp_weight) * gdp) + (max(0.0, interest_weight) * rate))
    multiplier = 1.0 - intensity

    lo = min(max_multiplier, min_multiplier)
    hi = max(min_multiplier, max_multiplier)
    return float(np.clip(multiplier, lo, hi))


def build_recovery_multiplier_curve(
    *,
    horizon: int,
    shock_duration_quarters: int,
    trough_multiplier: float,
    recovery_curve: RecoveryCurve | str,
    profile: RecessionCurveProfile,
) -> np.ndarray:
    """
    Build period-level multipliers for shock and recovery phases.
    """

    h = max(0, int(horizon))
    if h == 0:
        return np.array([], dtype=np.float64)

    shock_len = min(h, max(0, int(shock_duration_quarters)))
    multipliers = np.ones(h, dtype=np.float64)
    if shock_len == 0:
        return multipliers

    multipliers[:shock_len] = np.linspace(1.0, trough_multiplier, num=shock_len, dtype=np.float64)

    remaining = h - shock_len
    if remaining <= 0:
        return multipliers

    curve = str(recovery_curve or "u_shape").strip().lower()
    target_ratio = max(0.0, float(profile.target_ratio))
    if curve in {"v_shape", "u_shape"}:
        target_ratio = max(1.0, target_ratio)

    plateau = min(remaining, max(0, int(profile.plateau_quarters)))
    recovery = max(1, int(profile.recovery_quarters))

    tail = np.full(remaining, trough_multiplier, dtype=np.float64)
    if plateau < remaining:
        ramp_points = remaining - plateau
        steps = np.arange(1, ramp_points + 1, dtype=np.float64)
        progression = np.minimum(1.0, steps / float(recovery))
        tail[plateau:] = trough_multiplier + ((target_ratio - trough_multiplier) * progression)
        if curve == "l_shape":
            tail[plateau:] = np.minimum(tail[plateau:], target_ratio)

    multipliers[shock_len:] = tail
    return multipliers


def model_recession_projection(
    base_projected_revenue: Sequence[Any] | np.ndarray,
    *,
    gdp_contraction_rate: Any | None = None,
    interest_rate_spike: Any | None = None,
    industry_sensitivity_coefficient: Any = 1.0,
    shock_duration_quarters: int = 2,
    recovery_curve: RecoveryCurve | str | None = None,
    macro_metric_rows: Sequence[Mapping[str, Any]] = (),
    country_code: str | None = None,
    config: RecessionModelConfig | None = None,
) -> dict[str, Any]:
    """
    Deterministic recession impact modeling for projected revenue trajectories.
    """

    cfg = config or RecessionModelConfig()
    base = _to_float_array(base_projected_revenue)
    finite_base = np.isfinite(base)
    if base.size == 0 or not np.any(finite_base):
        return {
            "shock_phase_projection": [],
            "recovery_projection": [],
            "net_revenue_impact": 0.0,
            "recovery_time_estimate": "no_horizon",
            "revenue_shock_multiplier": None,
        }

    derived = extract_macro_shock_inputs(
        macro_metric_rows,
        country_code=country_code,
        percent_threshold=cfg.percent_threshold,
        percent_ceiling=cfg.percent_ceiling,
    )

    resolved_gdp = gdp_contraction_rate if gdp_contraction_rate is not None else derived.get("gdp_contraction_rate")
    resolved_rate = interest_rate_spike if interest_rate_spike is not None else derived.get("interest_rate_spike")

    shock_multiplier = compute_revenue_shock_multiplier(
        gdp_contraction_rate=resolved_gdp or 0.0,
        interest_rate_spike=resolved_rate or 0.0,
        industry_sensitivity_coefficient=industry_sensitivity_coefficient,
        gdp_weight=cfg.gdp_weight,
        interest_weight=cfg.interest_weight,
        min_multiplier=cfg.min_shock_multiplier,
        max_multiplier=cfg.max_shock_multiplier,
    )

    curve = str(recovery_curve or cfg.default_recovery_curve).strip().lower() or "u_shape"
    if curve not in {"v_shape", "u_shape", "l_shape"}:
        curve = "u_shape"
    profile = cfg.curve_profile(curve)

    multipliers = build_recovery_multiplier_curve(
        horizon=base.size,
        shock_duration_quarters=shock_duration_quarters,
        trough_multiplier=shock_multiplier,
        recovery_curve=curve,
        profile=profile,
    )

    adjusted = np.where(np.isfinite(base), base * multipliers, np.nan)
    shock_len = min(base.size, max(0, int(shock_duration_quarters)))

    net_impact = _net_impact_percent(base, adjusted, zero_guard=cfg.zero_guard)
    recovery_time = _estimate_recovery_time(
        base=base,
        adjusted=adjusted,
        threshold=max(0.0, float(cfg.recovery_threshold)),
        start_index=shock_len,
    )

    return {
        "shock_phase_projection": _to_optional_list(adjusted[:shock_len]),
        "recovery_projection": _to_optional_list(adjusted[shock_len:]),
        "net_revenue_impact": net_impact,
        "recovery_time_estimate": recovery_time,
        "revenue_shock_multiplier": round(float(shock_multiplier), 6),
    }


def _filter_macro_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    country_code: str | None,
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    normalized_country = str(country_code or "").strip().upper()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if normalized_country:
            row_country = str(row.get("country_code") or "").strip().upper()
            if row_country and row_country != normalized_country:
                continue
        selected.append(row)
    return selected


def _series_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    percent_threshold: float,
    percent_ceiling: float,
) -> list[tuple[date, float]]:
    out: list[tuple[date, float]] = []
    for row in rows:
        period_end = _parse_date_like(row.get("period_end") or row.get("timestamp"))
        value = _coerce_float(row.get("value", row.get("metric_value")))
        if period_end is None or value is None:
            continue
        out.append((period_end, _normalize_rate(value, percent_threshold=percent_threshold, percent_ceiling=percent_ceiling)))
    out.sort(key=lambda item: item[0])
    return out


def _derive_gdp_contraction(series: Sequence[tuple[date, float]]) -> float | None:
    if len(series) < 2:
        return None
    previous = series[-2][1]
    latest = series[-1][1]
    denominator = max(abs(previous), 1e-12)
    growth = (latest - previous) / denominator
    return max(0.0, -growth)


def _derive_interest_rate_spike(series: Sequence[tuple[date, float]]) -> float | None:
    if len(series) < 2:
        return None
    delta = series[-1][1] - series[-2][1]
    return max(0.0, delta)


def _is_gdp_metric(name: str) -> bool:
    normalized = name.strip().lower()
    return normalized in {"gdp", "gdp_growth", "gdp_growth_rate"}


def _is_policy_rate_metric(name: str) -> bool:
    normalized = name.strip().lower()
    return normalized in {"policy_rate", "interest_rate", "policy_interest_rate"}


def _normalize_rate(
    value: Any,
    *,
    percent_threshold: float = 1.0,
    percent_ceiling: float = 500.0,
) -> float:
    parsed = _coerce_float(value)
    if parsed is None:
        return 0.0
    if abs(parsed) >= float(percent_threshold) and abs(parsed) <= float(percent_ceiling):
        return float(parsed / 100.0)
    return float(parsed)


def _estimate_recovery_time(
    *,
    base: np.ndarray,
    adjusted: np.ndarray,
    threshold: float,
    start_index: int,
) -> str:
    if base.size == 0:
        return "no_horizon"
    valid = np.isfinite(base) & np.isfinite(adjusted) & (np.abs(base) > 1e-12)
    if not np.any(valid):
        return "insufficient_data"
    ratio = np.full(base.shape, np.nan, dtype=np.float64)
    ratio[valid] = adjusted[valid] / base[valid]
    start = max(0, min(int(start_index), ratio.size))
    recovered = np.where(np.isfinite(ratio[start:]) & (ratio[start:] >= threshold))[0]
    if recovered.size == 0:
        return "not_recovered_within_horizon"
    return f"{int(recovered[0]) + start + 1}Q"


def _net_impact_percent(base: np.ndarray, adjusted: np.ndarray, *, zero_guard: float) -> float:
    base_sum = float(np.nansum(base))
    adjusted_sum = float(np.nansum(adjusted))
    denominator = max(abs(base_sum), max(1e-12, float(zero_guard)))
    return round(((adjusted_sum - base_sum) / denominator) * 100.0, 6)


def _to_float_array(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray):
        if np.issubdtype(values.dtype, np.number):
            return values.astype(np.float64, copy=False).ravel()
        raw: Sequence[Any] = values.tolist()
    else:
        raw = values

    out = np.full(len(raw), np.nan, dtype=np.float64)
    for idx, value in enumerate(raw):
        parsed = _coerce_float(value)
        if parsed is not None:
            out[idx] = parsed
    return out


def _parse_date_like(value: Any) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00")
    try:
        return date.fromisoformat(normalized[:10])
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        return None


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


def _round_or_none(value: Any) -> float | None:
    parsed = _coerce_float(value)
    if parsed is None:
        return None
    return round(float(parsed), 6)


def _to_optional_list(values: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for value in values.astype(np.float64, copy=False):
        if np.isfinite(value):
            out.append(round(float(value), 6))
        else:
            out.append(None)
    return out
