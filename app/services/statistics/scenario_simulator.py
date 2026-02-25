from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
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


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


_SCENARIO_RULES = _as_dict(_load_business_rules().get("scenario_simulator"))
_SCENARIO_DEFAULTS = _as_dict(_SCENARIO_RULES.get("defaults"))
_SCENARIO_SHOCKS = _as_dict(_SCENARIO_RULES.get("shocks"))


@dataclass(frozen=True)
class ScenarioShock:
    growth_delta: float
    level_delta: float
    volatility_multiplier: float
    confidence_impact: float


@dataclass(frozen=True)
class ScenarioConfig:
    projection_periods: int = _as_int(_SCENARIO_DEFAULTS.get("projection_periods"), 1)
    min_confidence: float = _as_float(_SCENARIO_DEFAULTS.get("min_confidence"), 0.20)
    max_confidence: float = _as_float(_SCENARIO_DEFAULTS.get("max_confidence"), 1.0)


def simulate_deterministic_scenarios(
    metric_series: Mapping[str, Sequence[Any]],
    *,
    growth_context: Mapping[str, Any] | None,
    statistical_context: Mapping[str, Any] | None,
    multivariate_context: Mapping[str, Any] | None,
    preferred_metric_candidates: Sequence[str] = (),
) -> dict[str, Any]:
    cfg = ScenarioConfig()
    primary_metric = _pick_primary_metric(metric_series, growth_context, preferred_metric_candidates)
    latest_value = _latest_value(metric_series.get(primary_metric, ())) if primary_metric else None

    horizons = _extract_horizons(growth_context)
    base_growth = _pick_base_growth(horizons)
    anomaly_penalty = _anomaly_penalty(statistical_context)
    correlation_penalty = _correlation_penalty(multivariate_context)
    base_confidence = _base_confidence(growth_context, statistical_context, multivariate_context)

    scenarios: dict[str, Any] = {}
    warnings: list[str] = []
    if latest_value is None:
        warnings.append("No latest value available for scenario projections.")

    for scenario_name in ("best", "base", "worst"):
        shock = _shock_for_name(scenario_name)
        scenario_payload = _build_scenario(
            latest_value=latest_value,
            base_growth=base_growth,
            horizons=horizons,
            anomaly_penalty=anomaly_penalty,
            correlation_penalty=correlation_penalty,
            base_confidence=base_confidence,
            shock=shock,
            projection_periods=cfg.projection_periods,
            min_confidence=cfg.min_confidence,
            max_confidence=cfg.max_confidence,
        )
        scenarios[scenario_name] = scenario_payload

    worst = scenarios.get("worst", {})
    impact = {
        "confidence_delta_best_vs_base": _delta_confidence(scenarios.get("best"), scenarios.get("base")),
        "confidence_delta_worst_vs_base": _delta_confidence(worst, scenarios.get("base")),
    }

    insufficient = _as_dict(horizons.get("insufficient_history"))
    status = "partial" if any(bool(v) for v in insufficient.values()) or latest_value is None else "success"

    assumptions = {
        "projection_periods": cfg.projection_periods,
        "base_growth_selected": _round_or_none(base_growth),
        "anomaly_penalty": _round_or_none(anomaly_penalty),
        "correlation_penalty": _round_or_none(correlation_penalty),
        "shocks": {
            name: _shock_to_payload(_shock_for_name(name))
            for name in ("best", "base", "worst")
        },
    }

    return {
        "status": status,
        "primary_metric": primary_metric,
        "latest_value": _round_or_none(latest_value),
        "base_confidence": _round_or_none(base_confidence),
        "insufficient_history": {str(k): bool(v) for k, v in insufficient.items()},
        "warnings": warnings,
        "scenarios": scenarios,
        "metadata": {
            "assumptions": assumptions,
            "confidence_impact": impact,
        },
    }


def _build_scenario(
    *,
    latest_value: float | None,
    base_growth: float | None,
    horizons: Mapping[str, Any],
    anomaly_penalty: float,
    correlation_penalty: float,
    base_confidence: float,
    shock: ScenarioShock,
    projection_periods: int,
    min_confidence: float,
    max_confidence: float,
) -> dict[str, Any]:
    short = _as_optional_float(horizons.get("short_growth"))
    mid = _as_optional_float(horizons.get("mid_growth"))
    long = _as_optional_float(horizons.get("long_growth"))
    trend = _as_optional_float(horizons.get("trend_acceleration"))

    effective_base = base_growth if base_growth is not None else 0.0
    volatility_drag = anomaly_penalty * shock.volatility_multiplier
    projected_growth = effective_base + shock.growth_delta - volatility_drag

    short_proj = _apply_growth(short, shock.growth_delta, volatility_drag)
    mid_proj = _apply_growth(mid, shock.growth_delta, volatility_drag)
    long_proj = _apply_growth(long, shock.growth_delta, volatility_drag)
    trend_proj = _apply_growth(trend, shock.growth_delta / 2.0, volatility_drag / 2.0)

    projected_value = None
    if latest_value is not None:
        growth_factor = (1.0 + projected_growth + shock.level_delta)
        projected_value = latest_value * (growth_factor ** projection_periods)

    confidence = base_confidence + shock.confidence_impact - (correlation_penalty * 0.5)
    confidence = max(min_confidence, min(max_confidence, confidence))

    return {
        "projected_growth": _round_or_none(projected_growth),
        "projected_value": _round_or_none(projected_value),
        "horizon_growth": {
            "short": _round_or_none(short_proj),
            "mid": _round_or_none(mid_proj),
            "long": _round_or_none(long_proj),
            "trend_acceleration": _round_or_none(trend_proj),
        },
        "assumptions": _shock_to_payload(shock),
        "confidence_score": _round_or_none(confidence),
    }


def _pick_primary_metric(
    metric_series: Mapping[str, Sequence[Any]],
    growth_context: Mapping[str, Any] | None,
    preferred_metric_candidates: Sequence[str],
) -> str | None:
    if isinstance(growth_context, Mapping):
        name = str(growth_context.get("primary_metric") or "").strip()
        if name in metric_series:
            return name

    preferred = [_normalize(v) for v in preferred_metric_candidates if _normalize(v)]
    for candidate in preferred:
        for metric in metric_series:
            if _normalize(metric) == candidate:
                return metric

    best = None
    best_len = -1
    for metric, values in metric_series.items():
        length = len(coerce_numeric_series(values))
        if length > best_len:
            best = metric
            best_len = length
    return best


def _extract_horizons(growth_context: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(growth_context, Mapping):
        return {}
    horizons = growth_context.get("primary_horizons")
    if isinstance(horizons, Mapping):
        return dict(horizons)
    return {}


def _pick_base_growth(horizons: Mapping[str, Any]) -> float | None:
    for key in ("mid_growth", "short_growth", "long_growth"):
        value = _as_optional_float(horizons.get(key))
        if value is not None:
            return value
    return None


def _latest_value(values: Sequence[Any]) -> float | None:
    series = coerce_numeric_series(values)
    if not series:
        return None
    return series[-1]


def _anomaly_penalty(statistical_context: Mapping[str, Any] | None) -> float:
    if not isinstance(statistical_context, Mapping):
        return 0.0
    summary = statistical_context.get("anomaly_summary")
    if not isinstance(summary, Mapping):
        return 0.0
    count = summary.get("total_anomaly_points")
    try:
        value = float(count)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(0.10, value * 0.005))


def _correlation_penalty(multivariate_context: Mapping[str, Any] | None) -> float:
    if not isinstance(multivariate_context, Mapping):
        return 0.0
    corr = multivariate_context.get("correlation")
    if not isinstance(corr, Mapping):
        return 0.0
    total_pairs = _as_optional_float(corr.get("total_pairs")) or 0.0
    significant_pairs = _as_optional_float(corr.get("significant_pairs")) or 0.0
    if total_pairs <= 0.0:
        return 0.0
    missing_share = max(0.0, min(1.0, (total_pairs - significant_pairs) / total_pairs))
    return round(missing_share * 0.2, 6)


def _base_confidence(
    growth_context: Mapping[str, Any] | None,
    statistical_context: Mapping[str, Any] | None,
    multivariate_context: Mapping[str, Any] | None,
) -> float:
    values: list[float] = []
    for payload in (growth_context, statistical_context, multivariate_context):
        if not isinstance(payload, Mapping):
            continue
        confidence = _as_optional_float(payload.get("confidence_score"))
        if confidence is not None:
            values.append(confidence)
    if not values:
        return 0.6
    return sum(values) / len(values)


def _shock_for_name(name: str) -> ScenarioShock:
    shock_map = _as_dict(_SCENARIO_SHOCKS.get(name))
    return ScenarioShock(
        growth_delta=_as_float(shock_map.get("growth_delta"), 0.0),
        level_delta=_as_float(shock_map.get("level_delta"), 0.0),
        volatility_multiplier=_as_float(shock_map.get("volatility_multiplier"), 1.0),
        confidence_impact=_as_float(shock_map.get("confidence_impact"), 0.0),
    )


def _shock_to_payload(shock: ScenarioShock) -> dict[str, float]:
    return {
        "growth_delta": round(shock.growth_delta, 6),
        "level_delta": round(shock.level_delta, 6),
        "volatility_multiplier": round(shock.volatility_multiplier, 6),
        "confidence_impact": round(shock.confidence_impact, 6),
    }


def _delta_confidence(s1: Any, s2: Any) -> float | None:
    if not isinstance(s1, Mapping) or not isinstance(s2, Mapping):
        return None
    c1 = _as_optional_float(s1.get("confidence_score"))
    c2 = _as_optional_float(s2.get("confidence_score"))
    if c1 is None or c2 is None:
        return None
    return round(c1 - c2, 6)


def _apply_growth(base: float | None, shock: float, penalty: float) -> float | None:
    if base is None:
        return None
    return base + shock - penalty


def _as_optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _round_or_none(value: Any) -> float | None:
    parsed = _as_optional_float(value)
    if parsed is None:
        return None
    return round(parsed, 6)

