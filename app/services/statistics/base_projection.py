from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal, Mapping, Sequence

import numpy as np

ProjectionMethod = Literal["cagr", "rolling_avg"]


@dataclass(frozen=True)
class BaseProjectionConfig:
    """
    Configuration for deterministic base-case revenue projections.

    Economic reasoning:
    - `client_weight` captures persistence of the client's own historical trend.
    - `industry_weight` anchors the projection to sector conditions.
    - `gdp_weight` adds broad macro-cycle context.
    """

    method: ProjectionMethod = "cagr"
    horizon_quarters: int = 4
    rolling_window: int = 4
    client_weight: float = 0.6
    industry_weight: float = 0.3
    gdp_weight: float = 0.1
    zero_guard: float = 1e-9


def compute_growth_rates(revenue: Sequence[Any] | np.ndarray, *, zero_guard: float = 1e-9) -> np.ndarray:
    """
    Compute period-over-period growth from a revenue series.

    Formula:
    growth[t] = (revenue[t] - revenue[t-1]) / max(abs(revenue[t-1]), zero_guard)
    """

    arr = _to_float_array(revenue)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    if arr.size <= 1:
        return out

    previous = arr[:-1]
    current = arr[1:]
    denom = np.maximum(np.abs(previous), max(float(zero_guard), 1e-12))
    valid = np.isfinite(previous) & np.isfinite(current) & (denom > 0.0)
    growth = np.full(previous.shape, np.nan, dtype=np.float64)
    growth[valid] = (current[valid] - previous[valid]) / denom[valid]
    out[1:] = growth
    return out


def cagr_growth(revenue: Sequence[Any] | np.ndarray, *, zero_guard: float = 1e-9) -> float | None:
    """
    Compute simple CAGR from historical revenue.

    Formula:
    CAGR = (R_end / R_start)^(1 / n_periods) - 1
    """

    arr = _to_float_array(revenue)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return None
    start = float(finite[0])
    end = float(finite[-1])
    periods = int(finite.size - 1)
    if periods <= 0 or start <= max(float(zero_guard), 1e-12) or end <= 0.0:
        return None
    return float((end / start) ** (1.0 / periods) - 1.0)


def rolling_average_growth(
    revenue: Sequence[Any] | np.ndarray,
    *,
    window: int = 4,
    zero_guard: float = 1e-9,
) -> float | None:
    """
    Compute projection growth from the rolling mean of recent growth rates.

    Formula:
    rolling_avg_growth = mean(last `window` valid period growth values)
    """

    growth = compute_growth_rates(revenue, zero_guard=zero_guard)
    valid = growth[np.isfinite(growth)]
    if valid.size == 0:
        return None
    w = max(1, int(window))
    tail = valid[-w:]
    return float(np.mean(tail))


def build_projection_periods(
    *,
    horizon_quarters: int,
    historical_period_ends: Sequence[Any] | None = None,
) -> np.ndarray:
    """
    Build future quarter-end dates for projection horizon.

    If historical period ends are present, future periods continue from the
    last known period end. Otherwise a NaT-filled array is returned.
    """

    horizon = max(0, int(horizon_quarters))
    if horizon == 0:
        return np.array([], dtype="datetime64[D]")

    if not historical_period_ends:
        return np.array([np.datetime64("NaT", "D")] * horizon, dtype="datetime64[D]")

    hist = _to_datetime64_day(historical_period_ends)
    if hist.size == 0 or np.isnat(hist[-1]):
        return np.array([np.datetime64("NaT", "D")] * horizon, dtype="datetime64[D]")

    last_date = _datetime64_to_date(hist[-1])
    if last_date is None:
        return np.array([np.datetime64("NaT", "D")] * horizon, dtype="datetime64[D]")

    out: list[np.datetime64] = []
    year = last_date.year
    month = last_date.month
    for _ in range(horizon):
        year, month = _add_months(year, month, 3)
        q_end_month = month
        q_end_day = _month_end_day(year, q_end_month)
        out.append(np.datetime64(date(year, q_end_month, q_end_day), "D"))
    return np.array(out, dtype="datetime64[D]")


def align_growth_to_periods(
    projection_period_ends: Sequence[Any] | np.ndarray,
    *,
    scalar_growth: float | None = None,
    growth_rows: Sequence[Mapping[str, Any]] | None = None,
    value_key: str = "value",
) -> np.ndarray:
    """
    Align external growth rates (industry/GDP) to projection periods.

    Priority:
    1. interval-based rows using containment:
       period_start <= projection_period_end <= period_end
    2. scalar growth fallback broadcast across all periods
    """

    periods = _to_datetime64_day(projection_period_ends)
    if periods.size == 0:
        return np.array([], dtype=np.float64)

    aligned = np.full(periods.shape, np.nan, dtype=np.float64)
    if growth_rows:
        starts, ends, values = _extract_interval_values(growth_rows, value_key=value_key)
        if starts.size > 0:
            order = np.argsort(starts.astype(np.int64), kind="stable")
            starts = starts[order]
            ends = ends[order]
            values = values[order]

            valid = ~np.isnat(periods)
            valid_periods = periods[valid]
            idx = np.searchsorted(starts, valid_periods, side="right") - 1
            has = idx >= 0
            if np.any(has):
                candidate_idx = idx[has]
                candidate_periods = valid_periods[has]
                within = candidate_periods <= ends[candidate_idx]
                target_positions = np.flatnonzero(valid)[has][within]
                aligned[target_positions] = values[candidate_idx[within]]

    if scalar_growth is not None:
        scalar = _coerce_float(scalar_growth)
        if scalar is not None:
            missing = ~np.isfinite(aligned)
            aligned[missing] = scalar

    return aligned


def blend_projection_growth(
    *,
    baseline_growth: float,
    industry_growth: np.ndarray,
    gdp_growth: np.ndarray,
    client_weight: float,
    industry_weight: float,
    gdp_weight: float,
) -> np.ndarray:
    """
    Blend client baseline with aligned industry/GDP trajectories.

    Formula per period `t`:
    g_t = (w_c * g_client + w_i * g_industry_t + w_g * g_gdp_t) / (w_c + w_i + w_g available)

    Missing macro terms are excluded from denominator for that period.
    """

    horizon = max(industry_growth.size, gdp_growth.size)
    if horizon == 0:
        return np.array([], dtype=np.float64)

    industry = _resize_array(industry_growth, horizon)
    gdp = _resize_array(gdp_growth, horizon)
    baseline = np.full(horizon, float(baseline_growth), dtype=np.float64)

    numerator = np.full(horizon, 0.0, dtype=np.float64)
    denominator = np.full(horizon, 0.0, dtype=np.float64)

    wc = max(0.0, float(client_weight))
    wi = max(0.0, float(industry_weight))
    wg = max(0.0, float(gdp_weight))

    if wc > 0.0:
        numerator += wc * baseline
        denominator += wc

    industry_valid = np.isfinite(industry)
    if wi > 0.0 and np.any(industry_valid):
        numerator[industry_valid] += wi * industry[industry_valid]
        denominator[industry_valid] += wi

    gdp_valid = np.isfinite(gdp)
    if wg > 0.0 and np.any(gdp_valid):
        numerator[gdp_valid] += wg * gdp[gdp_valid]
        denominator[gdp_valid] += wg

    out = np.full(horizon, np.nan, dtype=np.float64)
    valid = denominator > 0.0
    out[valid] = numerator[valid] / denominator[valid]
    return out


def project_revenue_from_growth(last_revenue: float, projected_growth: np.ndarray) -> np.ndarray:
    """
    Project revenue recursively from growth trajectory.

    Formula:
    Revenue_t = Revenue_{t-1} * (1 + growth_t)
    """

    if projected_growth.size == 0:
        return np.array([], dtype=np.float64)

    growth = projected_growth.astype(np.float64, copy=False)
    multipliers = np.where(np.isfinite(growth), 1.0 + growth, np.nan)
    out = np.full(growth.shape, np.nan, dtype=np.float64)

    running = float(last_revenue)
    if not np.isfinite(running):
        return out

    for idx, factor in enumerate(multipliers):
        if not np.isfinite(factor):
            out[idx] = np.nan
            continue
        running = running * factor
        out[idx] = running
    return out


def project_base_case(
    historical_revenue: Sequence[Any] | np.ndarray,
    *,
    industry_growth_rate: float | None = None,
    industry_growth_rows: Sequence[Mapping[str, Any]] | None = None,
    gdp_growth_rate: float | None = None,
    gdp_growth_rows: Sequence[Mapping[str, Any]] | None = None,
    historical_period_ends: Sequence[Any] | None = None,
    config: BaseProjectionConfig | None = None,
) -> dict[str, Any]:
    """
    Build base-case projection trajectories.

    Supports:
    - Simple CAGR baseline projection
    - Rolling-average-growth baseline projection

    Returns projected revenue and growth trajectories over `horizon_quarters`.
    """

    cfg = config or BaseProjectionConfig()
    history = _to_float_array(historical_revenue)
    history_finite = history[np.isfinite(history)]
    if history_finite.size == 0:
        raise ValueError("historical_revenue must contain at least one numeric value.")

    last_revenue = float(history_finite[-1])
    if cfg.method == "cagr":
        baseline = cagr_growth(history_finite, zero_guard=cfg.zero_guard)
        method_formula = "baseline_growth = (R_end / R_start)^(1 / n_periods) - 1"
    elif cfg.method == "rolling_avg":
        baseline = rolling_average_growth(
            history_finite,
            window=cfg.rolling_window,
            zero_guard=cfg.zero_guard,
        )
        method_formula = "baseline_growth = mean(last_window period growth rates)"
    else:
        raise ValueError(f"Unsupported projection method: {cfg.method}")

    if baseline is None:
        baseline = 0.0

    projection_periods = build_projection_periods(
        horizon_quarters=cfg.horizon_quarters,
        historical_period_ends=historical_period_ends,
    )
    horizon = projection_periods.size
    if horizon == 0:
        return {
            "status": "success",
            "method": cfg.method,
            "projection_horizon_quarters": 0,
            "projected_period_end": [],
            "projected_growth_rate": [],
            "projected_revenue": [],
            "assumptions": {
                "baseline_growth": round(float(baseline), 6),
            },
            "formulas": {
                "baseline": method_formula,
                "blended_growth": "(w_client*g_client + w_industry*g_industry + w_gdp*g_gdp) / available_weights",
                "revenue_projection": "Revenue_t = Revenue_{t-1} * (1 + growth_t)",
            },
        }

    industry = align_growth_to_periods(
        projection_periods,
        scalar_growth=industry_growth_rate,
        growth_rows=industry_growth_rows,
    )
    gdp = align_growth_to_periods(
        projection_periods,
        scalar_growth=gdp_growth_rate,
        growth_rows=gdp_growth_rows,
    )

    projected_growth = blend_projection_growth(
        baseline_growth=float(baseline),
        industry_growth=industry,
        gdp_growth=gdp,
        client_weight=cfg.client_weight,
        industry_weight=cfg.industry_weight,
        gdp_weight=cfg.gdp_weight,
    )
    projected_revenue = project_revenue_from_growth(last_revenue, projected_growth)

    return {
        "status": "success",
        "method": cfg.method,
        "projection_horizon_quarters": int(cfg.horizon_quarters),
        "projected_period_end": _datetime64_to_list(projection_periods),
        "projected_growth_rate": _to_optional_list(projected_growth),
        "projected_revenue": _to_optional_list(projected_revenue),
        "assumptions": {
            "baseline_growth": round(float(baseline), 6),
            "last_historical_revenue": round(last_revenue, 6),
            "weights": {
                "client_weight": round(float(cfg.client_weight), 6),
                "industry_weight": round(float(cfg.industry_weight), 6),
                "gdp_weight": round(float(cfg.gdp_weight), 6),
            },
            "industry_growth_aligned": _to_optional_list(industry),
            "gdp_growth_aligned": _to_optional_list(gdp),
        },
        "formulas": {
            "baseline": method_formula,
            "blended_growth": "(w_client*g_client + w_industry*g_industry + w_gdp*g_gdp) / available_weights",
            "revenue_projection": "Revenue_t = Revenue_{t-1} * (1 + growth_t)",
        },
    }


def _extract_interval_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts: list[np.datetime64] = []
    ends: list[np.datetime64] = []
    values: list[float] = []

    for row in rows:
        start_raw = row.get("period_start", row.get("timestamp"))
        end_raw = row.get("period_end", row.get("timestamp"))
        value_raw = row.get(value_key, row.get("growth_rate", row.get("metric_value", row.get("value"))))

        start = _parse_date_like(start_raw)
        end = _parse_date_like(end_raw)
        value = _coerce_float(value_raw)
        if start is None or end is None or value is None:
            continue
        starts.append(np.datetime64(start, "D"))
        ends.append(np.datetime64(end, "D"))
        values.append(value)

    if not starts:
        empty_dates = np.array([], dtype="datetime64[D]")
        return empty_dates, empty_dates, np.array([], dtype=np.float64)

    return (
        np.array(starts, dtype="datetime64[D]"),
        np.array(ends, dtype="datetime64[D]"),
        np.array(values, dtype=np.float64),
    )


def _to_float_array(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray):
        if np.issubdtype(values.dtype, np.number):
            return values.astype(np.float64, copy=False).ravel()
        sequence: Sequence[Any] = values.tolist()
    else:
        sequence = values

    out = np.full(len(sequence), np.nan, dtype=np.float64)
    for idx, value in enumerate(sequence):
        parsed = _coerce_float(value)
        if parsed is not None:
            out[idx] = parsed
    return out


def _to_datetime64_day(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[D]")
    parsed: list[np.datetime64] = []
    for value in values:
        as_date = _parse_date_like(value)
        parsed.append(np.datetime64(as_date, "D") if as_date is not None else np.datetime64("NaT", "D"))
    return np.array(parsed, dtype="datetime64[D]")


def _parse_date_like(value: Any) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        return date.fromisoformat(str(value.astype("datetime64[D]")))
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
    return float(parsed) if np.isfinite(parsed) else None


def _resize_array(values: np.ndarray, size: int) -> np.ndarray:
    if values.size == size:
        return values.astype(np.float64, copy=False)
    out = np.full(size, np.nan, dtype=np.float64)
    if values.size == 0:
        return out
    limit = min(size, values.size)
    out[:limit] = values[:limit]
    return out


def _add_months(year: int, month: int, delta: int) -> tuple[int, int]:
    total = (year * 12 + (month - 1)) + delta
    return (total // 12, (total % 12) + 1)


def _month_end_day(year: int, month: int) -> int:
    return int(calendar.monthrange(year, month)[1])


def _datetime64_to_date(value: np.datetime64) -> date | None:
    if np.isnat(value):
        return None
    return date.fromisoformat(str(value.astype("datetime64[D]")))


def _to_optional_list(values: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for item in values:
        if np.isfinite(item):
            out.append(round(float(item), 6))
        else:
            out.append(None)
    return out


def _datetime64_to_list(values: np.ndarray) -> list[str | None]:
    out: list[str | None] = []
    for item in values.astype("datetime64[D]"):
        if np.isnat(item):
            out.append(None)
        else:
            out.append(str(item))
    return out
