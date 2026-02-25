from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from app.services.statistics.base_projection import BaseProjectionConfig, project_base_case
from app.services.statistics.competitive_erosion import simulate_competitive_erosion
from app.services.statistics.market_share_simulation import simulate_market_share
from app.services.statistics.recession_modeling import model_recession_projection

ScenarioType = Literal["base", "erosion", "recession", "combined"]

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


_ORCH_RULES = _as_dict(_load_business_rules().get("scenario_orchestration"))
_ORCH_DEFAULTS = _as_dict(_ORCH_RULES.get("defaults"))
_ORCH_SCENARIOS = _as_dict(_ORCH_RULES.get("scenarios"))


@dataclass(frozen=True)
class ScenarioToggles:
    use_erosion: bool = False
    use_recession: bool = False
    use_market_share: bool = True


@dataclass(frozen=True)
class ScenarioOrchestrationConfig:
    downside_weight: float = _as_float(_ORCH_DEFAULTS.get("downside_weight"), 0.70)
    drawdown_weight: float = _as_float(_ORCH_DEFAULTS.get("drawdown_weight"), 0.30)
    upside_credit_weight: float = _as_float(_ORCH_DEFAULTS.get("upside_credit_weight"), 0.40)
    zero_guard: float = _as_float(_ORCH_DEFAULTS.get("zero_guard"), 1e-9)
    default_market_share_value: float = _as_float(_ORCH_DEFAULTS.get("default_market_share_value"), 0.0)

    def toggles_for(self, scenario_name: str) -> ScenarioToggles:
        payload = _as_dict(_ORCH_SCENARIOS.get(str(scenario_name or "").strip().lower()))
        if not payload:
            payload = _as_dict(_ORCH_SCENARIOS.get("base"))
        return ScenarioToggles(
            use_erosion=bool(payload.get("use_erosion", False)),
            use_recession=bool(payload.get("use_recession", False)),
            use_market_share=bool(payload.get("use_market_share", True)),
        )


def run_base_scenario(config: Mapping[str, Any]) -> dict[str, Any]:
    return orchestrate_scenario(scenario_type="base", config=config)


def run_erosion_scenario(config: Mapping[str, Any]) -> dict[str, Any]:
    return orchestrate_scenario(scenario_type="erosion", config=config)


def run_recession_scenario(config: Mapping[str, Any]) -> dict[str, Any]:
    return orchestrate_scenario(scenario_type="recession", config=config)


def run_combined_scenario(config: Mapping[str, Any]) -> dict[str, Any]:
    return orchestrate_scenario(scenario_type="combined", config=config)


def orchestrate_scenario(
    *,
    scenario_type: ScenarioType | str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Compose base projection, competitive erosion, recession modeling, and market share simulation.
    """

    scenario_name = _normalize_scenario_name(scenario_type)
    settings = ScenarioOrchestrationConfig()
    base_revenue = _run_base_projection_array(_as_dict(config.get("base_case")))
    toggles = _resolve_toggles(settings=settings, scenario_name=scenario_name, config=config)

    return _orchestrate_from_base(
        scenario_name=scenario_name,
        config=config,
        settings=settings,
        toggles=toggles,
        base_revenue=base_revenue,
    )


def orchestrate_all_scenarios(
    *,
    config: Mapping[str, Any],
    scenario_types: Sequence[ScenarioType | str] = ("base", "erosion", "recession", "combined"),
) -> dict[str, dict[str, Any]]:
    """
    Compute multiple scenarios while reusing one base projection.
    """

    settings = ScenarioOrchestrationConfig()
    base_revenue = _run_base_projection_array(_as_dict(config.get("base_case")))
    outputs: dict[str, dict[str, Any]] = {}
    for raw_name in scenario_types:
        scenario_name = _normalize_scenario_name(raw_name)
        toggles = _resolve_toggles(settings=settings, scenario_name=scenario_name, config=config)
        outputs[scenario_name] = _orchestrate_from_base(
            scenario_name=scenario_name,
            config=config,
            settings=settings,
            toggles=toggles,
            base_revenue=base_revenue,
        )
    return outputs


def orchestrate_scenarios_batch(
    requests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """
    Batch orchestration for cloud/multi-client execution.

    Request shape per item:
    - {"client_id": "...", "config": {...}, "scenario_type": "base"}
    - {"client_id": "...", "config": {...}, "scenario_types": ["base", "combined"]}
    """

    outputs: list[dict[str, Any]] = []
    for request in requests:
        if not isinstance(request, Mapping):
            continue
        cfg = request.get("config")
        if not isinstance(cfg, Mapping):
            continue
        client_id = request.get("client_id")
        scenario_types = request.get("scenario_types")
        if isinstance(scenario_types, Sequence) and not isinstance(scenario_types, (str, bytes, bytearray)):
            result = orchestrate_all_scenarios(config=cfg, scenario_types=tuple(scenario_types))
            outputs.append(
                {
                    "client_id": client_id,
                    "scenarios": result,
                }
            )
            continue

        scenario_type = request.get("scenario_type", "base")
        result = orchestrate_scenario(scenario_type=scenario_type, config=cfg)
        outputs.append(
            {
                "client_id": client_id,
                "scenario": result,
            }
        )
    return outputs


def _orchestrate_from_base(
    *,
    scenario_name: str,
    config: Mapping[str, Any],
    settings: ScenarioOrchestrationConfig,
    toggles: ScenarioToggles,
    base_revenue: np.ndarray,
) -> dict[str, Any]:
    if base_revenue.size == 0:
        return {
            "scenario_name": scenario_name,
            "revenue_projection": [],
            "market_share_projection": [],
            "risk_score": 0.0,
            "downside_risk_pct": 0.0,
            "upside_potential_pct": 0.0,
        }

    working_revenue = base_revenue.copy()
    erosion_share_series: np.ndarray | None = None

    if toggles.use_erosion:
        erosion_cfg = _as_dict(config.get("erosion"))
        erosion_output = simulate_competitive_erosion(
            current_market_share=erosion_cfg.get("current_market_share", settings.default_market_share_value),
            competitive_score=erosion_cfg.get("competitive_score", 50.0),
            industry_growth_rate=erosion_cfg.get("industry_growth_rate", 0.0),
            erosion_factor=erosion_cfg.get("erosion_factor", 0.0),
            horizon_periods=erosion_cfg.get("horizon_periods", int(working_revenue.size)),
            severity=str(erosion_cfg.get("severity", "moderate")),
            decay_mode=erosion_cfg.get("decay_mode"),
            seed=erosion_cfg.get("seed"),
            noise_std=erosion_cfg.get("noise_std"),
        )
        erosion_share_series = _to_share_fraction_array(erosion_output.get("market_share_series"))
        if erosion_share_series.size > 0:
            erosion_share_series = _resize_array(erosion_share_series, working_revenue.size)
            start_share = max(settings.zero_guard, _first_finite(erosion_share_series))
            erosion_factor_series = np.where(
                np.isfinite(erosion_share_series),
                erosion_share_series / start_share,
                np.nan,
            )
            erosion_factor_series = np.where(np.isfinite(erosion_factor_series), erosion_factor_series, 1.0)
            working_revenue = working_revenue * np.maximum(erosion_factor_series, 0.0)

    if toggles.use_recession:
        recession_cfg = _as_dict(config.get("recession"))
        recession_output = model_recession_projection(
            base_projected_revenue=working_revenue,
            gdp_contraction_rate=recession_cfg.get("gdp_contraction_rate"),
            interest_rate_spike=recession_cfg.get("interest_rate_spike"),
            industry_sensitivity_coefficient=recession_cfg.get("industry_sensitivity_coefficient", 1.0),
            shock_duration_quarters=int(recession_cfg.get("shock_duration_quarters", 2)),
            recovery_curve=recession_cfg.get("recovery_curve"),
            macro_metric_rows=recession_cfg.get("macro_metric_rows", ()) or (),
            country_code=recession_cfg.get("country_code"),
        )
        shock_projection = _to_float_array(recession_output.get("shock_phase_projection", []))
        recovery_projection = _to_float_array(recession_output.get("recovery_projection", []))
        if shock_projection.size and recovery_projection.size:
            recession_projection = np.concatenate((shock_projection, recovery_projection))
        elif shock_projection.size:
            recession_projection = shock_projection
        else:
            recession_projection = recovery_projection
        if recession_projection.size:
            working_revenue = _resize_array(recession_projection, working_revenue.size)

    market_share_projection: list[float] = []
    if toggles.use_market_share:
        market_cfg = _as_dict(config.get("market_share"))
        market_output = simulate_market_share(
            total_industry_size=market_cfg.get("total_industry_size", 0.0),
            client_revenue_projection=working_revenue,
            competitor_growth_rates=market_cfg.get("competitor_growth_rates"),
            new_entrant_factor=market_cfg.get("new_entrant_factor"),
            industry_expansion_rate=market_cfg.get("industry_expansion_rate", 0.0),
            scenario=market_cfg.get("scenario"),
        )
        market_share_projection = _clean_numeric_list(market_output.get("market_share_series"))
    elif erosion_share_series is not None and erosion_share_series.size:
        market_share_projection = [round(float(v), 6) for v in erosion_share_series.tolist()]

    risk = _compute_risk_metrics(
        base_revenue=base_revenue,
        scenario_revenue=working_revenue,
        downside_weight=settings.downside_weight,
        drawdown_weight=settings.drawdown_weight,
        upside_credit_weight=settings.upside_credit_weight,
        zero_guard=settings.zero_guard,
    )

    return {
        "scenario_name": scenario_name,
        "revenue_projection": _clean_numeric_list(working_revenue.tolist()),
        "market_share_projection": market_share_projection,
        "risk_score": risk["risk_score"],
        "downside_risk_pct": risk["downside_risk_pct"],
        "upside_potential_pct": risk["upside_potential_pct"],
    }


def _resolve_toggles(
    *,
    settings: ScenarioOrchestrationConfig,
    scenario_name: str,
    config: Mapping[str, Any],
) -> ScenarioToggles:
    toggles = settings.toggles_for(scenario_name)
    override_toggles = _as_dict(config.get("toggles"))
    if not override_toggles:
        return toggles
    return ScenarioToggles(
        use_erosion=bool(override_toggles.get("use_erosion", toggles.use_erosion)),
        use_recession=bool(override_toggles.get("use_recession", toggles.use_recession)),
        use_market_share=bool(override_toggles.get("use_market_share", toggles.use_market_share)),
    )


def _run_base_projection_array(base_config: Mapping[str, Any]) -> np.ndarray:
    precomputed = base_config.get("revenue_projection")
    if isinstance(precomputed, Sequence) and not isinstance(precomputed, (str, bytes, bytearray)):
        return _to_float_array(precomputed)

    historical_revenue = base_config.get("historical_revenue")
    if not isinstance(historical_revenue, Sequence) or isinstance(historical_revenue, (str, bytes, bytearray)):
        return np.array([], dtype=np.float64)

    projection_cfg_raw = _as_dict(base_config.get("projection_config"))
    projection_cfg = BaseProjectionConfig(
        method=str(projection_cfg_raw.get("method", "cagr")),
        horizon_quarters=int(projection_cfg_raw.get("horizon_quarters", 4)),
        rolling_window=int(projection_cfg_raw.get("rolling_window", 4)),
        client_weight=float(projection_cfg_raw.get("client_weight", 0.6)),
        industry_weight=float(projection_cfg_raw.get("industry_weight", 0.3)),
        gdp_weight=float(projection_cfg_raw.get("gdp_weight", 0.1)),
        zero_guard=float(projection_cfg_raw.get("zero_guard", 1e-9)),
    )

    projected = project_base_case(
        historical_revenue,
        industry_growth_rate=base_config.get("industry_growth_rate"),
        industry_growth_rows=base_config.get("industry_growth_rows"),
        gdp_growth_rate=base_config.get("gdp_growth_rate"),
        gdp_growth_rows=base_config.get("gdp_growth_rows"),
        historical_period_ends=base_config.get("historical_period_ends"),
        config=projection_cfg,
    )
    return _to_float_array(projected.get("projected_revenue", []))


def _compute_risk_metrics(
    *,
    base_revenue: np.ndarray,
    scenario_revenue: np.ndarray,
    downside_weight: float,
    drawdown_weight: float,
    upside_credit_weight: float,
    zero_guard: float,
) -> dict[str, float]:
    base = _resize_array(base_revenue, max(base_revenue.size, scenario_revenue.size))
    scenario = _resize_array(scenario_revenue, base.size)
    valid = np.isfinite(base) & np.isfinite(scenario)
    if not np.any(valid):
        return {
            "risk_score": 0.0,
            "downside_risk_pct": 0.0,
            "upside_potential_pct": 0.0,
        }

    base_sum = float(np.nansum(base[valid]))
    scenario_sum = float(np.nansum(scenario[valid]))
    denominator = max(abs(base_sum), max(1e-12, float(zero_guard)))

    downside = max(0.0, ((base_sum - scenario_sum) / denominator) * 100.0)
    upside = max(0.0, ((scenario_sum - base_sum) / denominator) * 100.0)

    per_period_den = np.maximum(np.abs(base[valid]), max(1e-12, float(zero_guard)))
    period_change = (scenario[valid] - base[valid]) / per_period_den
    worst_drawdown = max(0.0, -float(np.nanmin(period_change))) * 100.0 if period_change.size else 0.0

    risk_score = (
        max(0.0, downside_weight) * downside
        + max(0.0, drawdown_weight) * worst_drawdown
        - max(0.0, upside_credit_weight) * upside
    )
    risk_score = float(np.clip(risk_score, 0.0, 100.0))

    return {
        "risk_score": round(risk_score, 6),
        "downside_risk_pct": round(float(downside), 6),
        "upside_potential_pct": round(float(upside), 6),
    }


def _normalize_scenario_name(value: Any) -> str:
    normalized = str(value or "base").strip().lower()
    if normalized not in {"base", "erosion", "recession", "combined"}:
        return "base"
    return normalized


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


def _to_share_fraction_array(values: Any) -> np.ndarray:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return np.array([], dtype=np.float64)
    arr = _to_float_array(values)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.array([], dtype=np.float64)
    if np.nanmax(np.abs(finite)) > 1.0:
        arr = arr / 100.0
    return np.clip(arr, 0.0, 1.0)


def _resize_array(values: np.ndarray, size: int) -> np.ndarray:
    target = max(0, int(size))
    if target == 0:
        return np.array([], dtype=np.float64)
    if values.size == target:
        return values.astype(np.float64, copy=False)
    out = np.full(target, np.nan, dtype=np.float64)
    if values.size == 0:
        return out
    limit = min(target, values.size)
    out[:limit] = values[:limit]
    if values.size < target:
        out[limit:] = values[-1]
    return out


def _first_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(finite[0])


def _clean_numeric_list(values: Any) -> list[float]:
    if isinstance(values, np.ndarray):
        arr = values.astype(np.float64, copy=False).ravel()
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return []
        return np.round(finite, 6).astype(float).tolist()

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return []
    out: list[float] = []
    for item in values:
        parsed = _coerce_float(item)
        if parsed is None:
            continue
        out.append(round(float(parsed), 6))
    return out


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
