from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

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


_MARKET_RULES = _as_dict(_load_business_rules().get("market_share_simulation"))
_MARKET_DEFAULTS = _as_dict(_MARKET_RULES.get("defaults"))
_MARKET_SCENARIOS = _as_dict(_MARKET_RULES.get("scenario_assumptions"))


@dataclass(frozen=True)
class MarketScenarioAssumption:
    client_projection_multiplier: float = 1.0
    competitor_growth_multiplier: float = 1.0
    industry_expansion_multiplier: float = 1.0
    entrant_multiplier: float = 1.0


@dataclass(frozen=True)
class MarketShareSimulationConfig:
    default_scenario: str = str(_MARKET_DEFAULTS.get("default_scenario", "base")).strip().lower() or "base"
    default_new_entrant_factor: float = _as_float(_MARKET_DEFAULTS.get("new_entrant_factor"), 0.0)
    client_initial_share_fallback: float = _as_float(_MARKET_DEFAULTS.get("client_initial_share_fallback"), 0.10)
    percent_threshold: float = _as_float(_MARKET_DEFAULTS.get("percent_threshold"), 1.0)
    percent_ceiling: float = _as_float(_MARKET_DEFAULTS.get("percent_ceiling"), 500.0)
    rank_weight: float = _as_float(_MARKET_DEFAULTS.get("rank_weight"), 0.5)
    share_weight: float = _as_float(_MARKET_DEFAULTS.get("share_weight"), 0.5)
    zero_guard: float = _as_float(_MARKET_DEFAULTS.get("zero_guard"), 1e-9)

    def scenario(self, scenario_name: str | None) -> MarketScenarioAssumption:
        key = str(scenario_name or self.default_scenario).strip().lower() or self.default_scenario
        payload = _as_dict(_MARKET_SCENARIOS.get(key))
        if not payload:
            payload = _as_dict(_MARKET_SCENARIOS.get(self.default_scenario))
        return MarketScenarioAssumption(
            client_projection_multiplier=_as_float(payload.get("client_projection_multiplier"), 1.0),
            competitor_growth_multiplier=_as_float(payload.get("competitor_growth_multiplier"), 1.0),
            industry_expansion_multiplier=_as_float(payload.get("industry_expansion_multiplier"), 1.0),
            entrant_multiplier=_as_float(payload.get("entrant_multiplier"), 1.0),
        )


def simulate_market_share(
    *,
    total_industry_size: Any,
    client_revenue_projection: Sequence[Any] | np.ndarray,
    competitor_growth_rates: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    new_entrant_factor: Any | None = None,
    industry_expansion_rate: Any = 0.0,
    scenario: str | None = None,
    config: MarketShareSimulationConfig | None = None,
) -> dict[str, Any]:
    """
    Deterministic market-share simulation with multi-competitor support.
    """

    cfg = config or MarketShareSimulationConfig()
    assumption = cfg.scenario(scenario)

    client = _to_float_array(client_revenue_projection)
    horizon = int(client.size)
    if horizon == 0:
        return {
            "market_share_series": [],
            "relative_rank_series": [],
            "share_delta": 0.0,
            "competitive_shift_index": 0.0,
        }

    client = np.where(
        np.isfinite(client),
        np.maximum(client, 0.0) * max(0.0, assumption.client_projection_multiplier),
        np.nan,
    )
    if not np.any(np.isfinite(client)):
        return {
            "market_share_series": [],
            "relative_rank_series": [],
            "share_delta": 0.0,
            "competitive_shift_index": 0.0,
        }

    total_size = max(0.0, _coerce_float(total_industry_size) or 0.0)
    expansion = _to_rate_array(
        industry_expansion_rate,
        horizon=horizon,
        percent_threshold=cfg.percent_threshold,
        percent_ceiling=cfg.percent_ceiling,
    )
    expansion = expansion * max(0.0, assumption.industry_expansion_multiplier)
    expansion = np.where(np.isfinite(expansion), expansion, 0.0)
    industry_size = total_size * np.cumprod(1.0 + expansion, dtype=np.float64)

    competitor_growth_matrix = _parse_competitor_growth_rates(
        competitor_growth_rates,
        horizon=horizon,
        percent_threshold=cfg.percent_threshold,
        percent_ceiling=cfg.percent_ceiling,
    )
    competitor_growth_matrix = competitor_growth_matrix * max(0.0, assumption.competitor_growth_multiplier)

    initial_client = max(0.0, _first_finite(client))
    initial_industry = max(total_size, initial_client)
    if initial_industry <= 0.0:
        fallback_share = max(cfg.zero_guard, min(1.0, cfg.client_initial_share_fallback))
        initial_industry = initial_client / fallback_share if initial_client > 0.0 else 1.0

    initial_competitor_pool = max(0.0, initial_industry - initial_client)
    competitor_count = int(competitor_growth_matrix.shape[0])
    competitor_revenue = np.zeros((competitor_count, horizon), dtype=np.float64)
    if competitor_count > 0 and initial_competitor_pool > 0.0:
        initial_split = np.full(competitor_count, initial_competitor_pool / float(competitor_count), dtype=np.float64)
        competitor_revenue = initial_split[:, None] * np.cumprod(1.0 + competitor_growth_matrix, axis=1, dtype=np.float64)

    entrant_factor_raw = cfg.default_new_entrant_factor if new_entrant_factor is None else (_coerce_float(new_entrant_factor) or 0.0)
    entrant_factor = max(0.0, _normalize_rate(
        entrant_factor_raw,
        percent_threshold=cfg.percent_threshold,
        percent_ceiling=cfg.percent_ceiling,
    ))
    entrant_factor = entrant_factor * max(0.0, assumption.entrant_multiplier)
    entrant_revenue = industry_size * entrant_factor
    entrant_active = bool(np.any(np.isfinite(entrant_revenue) & (entrant_revenue > 0.0)))

    competitor_total = competitor_revenue.sum(axis=0) + entrant_revenue
    ecosystem_total = client + competitor_total
    market_denominator = np.maximum(industry_size, ecosystem_total)
    market_denominator = np.where(np.isfinite(market_denominator) & (market_denominator > cfg.zero_guard), market_denominator, np.nan)
    market_share = np.where(np.isfinite(client), client / market_denominator, np.nan)

    client_rank = _compute_client_rank(
        client=client,
        competitor_revenue=competitor_revenue,
        entrant_revenue=entrant_revenue,
        entrant_active=entrant_active,
    )

    first_share = _first_finite(market_share)
    last_share = _last_finite(market_share)
    share_delta_pp = (last_share - first_share) * 100.0

    first_rank = int(client_rank[0]) if client_rank.size else 1
    last_rank = int(client_rank[-1]) if client_rank.size else first_rank
    participants = max(1, competitor_count + (1 if entrant_active else 0))
    rank_change = (first_rank - last_rank) / float(participants)
    share_change = (last_share - first_share) / max(abs(first_share), cfg.zero_guard)
    shift_index = (
        max(0.0, cfg.rank_weight) * rank_change
        + max(0.0, cfg.share_weight) * share_change
    )

    return {
        "market_share_series": _to_optional_list(market_share),
        "relative_rank_series": [int(value) for value in client_rank.tolist()],
        "share_delta": round(float(share_delta_pp), 6),
        "competitive_shift_index": round(float(shift_index), 6),
    }


def _parse_competitor_growth_rates(
    growth_input: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    *,
    horizon: int,
    percent_threshold: float,
    percent_ceiling: float,
) -> np.ndarray:
    rows: list[np.ndarray] = []

    if isinstance(growth_input, Mapping):
        for raw_values in growth_input.values():
            rows.append(
                _to_rate_array(
                    raw_values,
                    horizon=horizon,
                    percent_threshold=percent_threshold,
                    percent_ceiling=percent_ceiling,
                )
            )
    elif isinstance(growth_input, Sequence):
        for item in growth_input:
            if not isinstance(item, Mapping):
                continue
            raw_values = item.get("growth_rates", item.get("growth_rate", item.get("rate", 0.0)))
            rows.append(
                _to_rate_array(
                    raw_values,
                    horizon=horizon,
                    percent_threshold=percent_threshold,
                    percent_ceiling=percent_ceiling,
                )
            )

    if not rows:
        return np.zeros((0, horizon), dtype=np.float64)
    return np.vstack(rows)


def _to_rate_array(
    values: Any,
    *,
    horizon: int,
    percent_threshold: float,
    percent_ceiling: float,
) -> np.ndarray:
    if isinstance(values, np.ndarray):
        arr = _to_float_array(values)
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        arr = _to_float_array(values)
    else:
        scalar = _normalize_rate(values, percent_threshold=percent_threshold, percent_ceiling=percent_ceiling)
        return np.full(horizon, scalar, dtype=np.float64)

    if arr.size == 0:
        return np.zeros(horizon, dtype=np.float64)

    arr = arr.astype(np.float64, copy=False)
    finite = np.isfinite(arr)
    percent_like = finite & (np.abs(arr) >= float(percent_threshold)) & (np.abs(arr) <= float(percent_ceiling))
    arr = arr.copy()
    arr[percent_like] = arr[percent_like] / 100.0
    if arr.size >= horizon:
        out = arr[:horizon]
    else:
        out = np.full(horizon, arr[-1], dtype=np.float64)
        out[:arr.size] = arr
    return np.where(np.isfinite(out), out, 0.0)


def _compute_client_rank(
    *,
    client: np.ndarray,
    competitor_revenue: np.ndarray,
    entrant_revenue: np.ndarray,
    entrant_active: bool,
) -> np.ndarray:
    """
    Compute client rank without full argsort matrix.

    Rank convention matches stable descending ordering used previously:
    equals do not outrank client, so comparisons are strictly `>`.
    """

    horizon = int(client.size)
    if horizon == 0:
        return np.array([], dtype=np.int64)

    client_cmp = np.where(np.isfinite(client), client, -np.inf)
    greater_count = np.zeros(horizon, dtype=np.int64)

    if competitor_revenue.size:
        comp_cmp = np.where(np.isfinite(competitor_revenue), competitor_revenue, -np.inf)
        greater_count += np.sum(comp_cmp > client_cmp, axis=0, dtype=np.int64)

    if entrant_active:
        entrant_cmp = np.where(np.isfinite(entrant_revenue), entrant_revenue, -np.inf)
        greater_count += (entrant_cmp > client_cmp).astype(np.int64, copy=False)

    return 1 + greater_count


def _normalize_rate(
    value: Any,
    *,
    percent_threshold: float,
    percent_ceiling: float,
) -> float:
    parsed = _coerce_float(value)
    if parsed is None:
        return 0.0
    if abs(parsed) >= float(percent_threshold) and abs(parsed) <= float(percent_ceiling):
        return float(parsed / 100.0)
    return float(parsed)


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


def _first_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(finite[0])


def _last_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(finite[-1])


def _to_optional_list(values: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for value in values.astype(np.float64, copy=False):
        if np.isfinite(value):
            out.append(round(float(value), 6))
        else:
            out.append(None)
    return out
