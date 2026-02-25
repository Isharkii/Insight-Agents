from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping, Sequence

import numpy as np


def compute_growth_from_levels(levels: Sequence[Any]) -> np.ndarray:
    """
    Compute nominal period-over-period growth from KPI levels.

    Formula:
    growth[t] = (level[t] - level[t-1]) / abs(level[t-1])

    Economic reasoning:
    This transforms raw business levels (for example revenue) into
    comparable relative changes. The denominator uses absolute prior level
    for stability when signs vary.
    """

    arr = _to_float_array(levels)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    if arr.size <= 1:
        return out

    previous = arr[:-1]
    current = arr[1:]
    denominator = np.abs(previous)
    valid = np.isfinite(previous) & np.isfinite(current) & (denominator > 0.0)
    growth = np.full(previous.shape, np.nan, dtype=np.float64)
    growth[valid] = (current[valid] - previous[valid]) / denominator[valid]
    out[1:] = growth
    return out


def normalize_rate_series(
    values: Sequence[Any] | np.ndarray,
    *,
    percent_threshold: float = 1.0,
    percent_ceiling: float = 500.0,
) -> np.ndarray:
    """
    Normalize mixed rate representations into decimal fractions.

    Values already in decimal form (for example 0.05) are preserved.
    Values likely expressed in percent points (for example 5 or 10) are
    scaled by 100 when abs(value) is above `percent_threshold`.
    """

    arr = _to_float_array(values)
    out = arr.copy()
    finite = np.isfinite(out)
    percent_like = finite & (np.abs(out) >= percent_threshold) & (np.abs(out) <= percent_ceiling)
    out[percent_like] = out[percent_like] / 100.0
    return out


def align_macro_series_to_periods(
    kpi_period_ends: Sequence[Any],
    macro_rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str = "value",
) -> np.ndarray:
    """
    Align macro values to KPI periods by interval containment.

    For each KPI period end, this selects the macro record where:
    period_start <= kpi_period_end <= period_end

    This supports monthly/quarterly macro series without requiring exact date
    equality. If no matching macro interval exists, output is NaN.
    """

    kpi_dates = _to_datetime64_day(kpi_period_ends)
    if len(macro_rows) == 0:
        return np.full(kpi_dates.shape, np.nan, dtype=np.float64)

    starts, ends, values = _extract_macro_arrays(macro_rows, value_key=value_key)
    if starts.size == 0:
        return np.full(kpi_dates.shape, np.nan, dtype=np.float64)

    # Search by period_start, then validate period_end containment.
    order = np.argsort(starts.astype("datetime64[D]").astype(np.int64), kind="stable")
    starts = starts[order]
    ends = ends[order]
    values = values[order]

    aligned = np.full(kpi_dates.shape, np.nan, dtype=np.float64)
    valid_kpi = ~np.isnat(kpi_dates)
    if not np.any(valid_kpi):
        return aligned

    kpi_valid_dates = kpi_dates[valid_kpi]
    indices = np.searchsorted(starts, kpi_valid_dates, side="right") - 1
    has_candidate = indices >= 0
    if not np.any(has_candidate):
        return aligned

    candidate_idx = indices[has_candidate]
    candidate_dates = kpi_valid_dates[has_candidate]
    within_bounds = candidate_dates <= ends[candidate_idx]

    target_positions = np.flatnonzero(valid_kpi)[has_candidate][within_bounds]
    aligned[target_positions] = values[candidate_idx[within_bounds]]
    return aligned


def compute_real_growth_rate(
    nominal_growth_rate: Sequence[Any] | np.ndarray,
    inflation_rate: Sequence[Any] | np.ndarray,
) -> np.ndarray:
    """
    Convert nominal growth into inflation-adjusted real growth.

    Formula:
    real_growth = ((1 + nominal_growth) / (1 + inflation)) - 1

    Economic reasoning:
    Nominal growth overstates performance when price levels rise quickly.
    Real growth removes broad inflation effects and better reflects volume/
    productivity-driven improvement.
    """

    nominal = _to_float_array(nominal_growth_rate)
    inflation = _to_float_array(inflation_rate)
    nominal, inflation = _broadcast_pair(nominal, inflation)

    denominator = 1.0 + inflation
    valid = np.isfinite(nominal) & np.isfinite(inflation) & (np.abs(denominator) > 1e-12)
    out = np.full(nominal.shape, np.nan, dtype=np.float64)
    out[valid] = ((1.0 + nominal[valid]) / denominator[valid]) - 1.0
    return out


def compute_growth_vs_gdp_delta(
    real_growth_rate: Sequence[Any] | np.ndarray,
    gdp_growth_rate: Sequence[Any] | np.ndarray,
) -> np.ndarray:
    """
    Measure business out/under-performance against macro GDP growth.

    Formula:
    growth_vs_gdp_delta = real_growth_rate - gdp_growth_rate

    Economic reasoning:
    Positive values indicate the business is growing faster than the broader
    economy after inflation adjustment.
    """

    real_growth = _to_float_array(real_growth_rate)
    gdp = _to_float_array(gdp_growth_rate)
    real_growth, gdp = _broadcast_pair(real_growth, gdp)

    out = np.full(real_growth.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(real_growth) & np.isfinite(gdp)
    out[valid] = real_growth[valid] - gdp[valid]
    return out


def compute_rate_adjusted_efficiency(
    real_growth_rate: Sequence[Any] | np.ndarray,
    policy_rate: Sequence[Any] | np.ndarray,
    *,
    min_policy_rate: float = -0.95,
) -> np.ndarray:
    """
    Discount real growth by policy-rate financing pressure.

    Formula:
    rate_adjusted_efficiency = real_growth_rate / (1 + max(policy_rate, min_policy_rate))

    Economic reasoning:
    Higher policy rates increase capital costs and the hurdle rate for
    growth quality. This metric scales real growth by that macro headwind.
    """

    real_growth = _to_float_array(real_growth_rate)
    rates = _to_float_array(policy_rate)
    real_growth, rates = _broadcast_pair(real_growth, rates)

    adjusted_rates = np.maximum(rates, min_policy_rate)
    denominator = 1.0 + adjusted_rates
    out = np.full(real_growth.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(real_growth) & np.isfinite(adjusted_rates) & (denominator > 1e-12)
    out[valid] = real_growth[valid] / denominator[valid]
    return out


def normalize_business_metrics_for_macro(
    *,
    kpi_period_ends: Sequence[Any],
    nominal_growth_rates: Sequence[Any] | None = None,
    kpi_levels: Sequence[Any] | None = None,
    inflation_rows: Sequence[Mapping[str, Any]] = (),
    gdp_rows: Sequence[Mapping[str, Any]] = (),
    policy_rate_rows: Sequence[Mapping[str, Any]] = (),
    round_digits: int = 6,
) -> dict[str, Any]:
    """
    End-to-end macro normalization for business growth metrics.

    Steps:
    1. Build or accept nominal growth series.
    2. Time-align inflation/GDP/policy-rate rows to KPI periods.
    3. Compute:
       - real growth rate
       - growth vs GDP delta
       - rate-adjusted efficiency
    """

    periods = _to_datetime64_day(kpi_period_ends)
    if nominal_growth_rates is None:
        if kpi_levels is None:
            raise ValueError("Provide either `nominal_growth_rates` or `kpi_levels`.")
        nominal = compute_growth_from_levels(kpi_levels)
    else:
        nominal = _to_float_array(nominal_growth_rates)

    if periods.size != nominal.size:
        raise ValueError("`kpi_period_ends` and growth series must have identical lengths.")

    nominal = normalize_rate_series(nominal)
    inflation = normalize_rate_series(
        align_macro_series_to_periods(periods, inflation_rows, value_key="value")
    )
    gdp = normalize_rate_series(
        align_macro_series_to_periods(periods, gdp_rows, value_key="value")
    )
    policy = normalize_rate_series(
        align_macro_series_to_periods(periods, policy_rate_rows, value_key="value")
    )

    real_growth = compute_real_growth_rate(nominal, inflation)
    vs_gdp = compute_growth_vs_gdp_delta(real_growth, gdp)
    efficiency = compute_rate_adjusted_efficiency(real_growth, policy)

    return {
        "formulas": {
            "real_growth_rate": "((1 + nominal_growth_rate) / (1 + inflation_rate)) - 1",
            "growth_vs_gdp_delta": "real_growth_rate - gdp_growth_rate",
            "rate_adjusted_efficiency": "real_growth_rate / (1 + policy_rate)",
        },
        "period_end": _datetime64_to_iso_list(periods),
        "nominal_growth_rate": _to_optional_list(nominal, round_digits=round_digits),
        "inflation_rate": _to_optional_list(inflation, round_digits=round_digits),
        "gdp_growth_rate": _to_optional_list(gdp, round_digits=round_digits),
        "policy_rate": _to_optional_list(policy, round_digits=round_digits),
        "real_growth_rate": _to_optional_list(real_growth, round_digits=round_digits),
        "growth_vs_gdp_delta": _to_optional_list(vs_gdp, round_digits=round_digits),
        "rate_adjusted_efficiency": _to_optional_list(efficiency, round_digits=round_digits),
    }


def _extract_macro_arrays(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts: list[np.datetime64] = []
    ends: list[np.datetime64] = []
    values: list[float] = []

    for row in rows:
        start_raw = row.get("period_start")
        end_raw = row.get("period_end")
        if start_raw is None:
            start_raw = row.get("timestamp")
        if end_raw is None:
            end_raw = row.get("timestamp")

        start_date = _parse_date_like(start_raw)
        end_date = _parse_date_like(end_raw)
        value = _coerce_float(row.get(value_key))

        if start_date is None or end_date is None or value is None:
            continue
        starts.append(np.datetime64(start_date, "D"))
        ends.append(np.datetime64(end_date, "D"))
        values.append(value)

    if not starts:
        empty_dates = np.array([], dtype="datetime64[D]")
        empty_values = np.array([], dtype=np.float64)
        return empty_dates, empty_dates, empty_values

    return (
        np.array(starts, dtype="datetime64[D]"),
        np.array(ends, dtype="datetime64[D]"),
        np.array(values, dtype=np.float64),
    )


def _to_datetime64_day(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[D]")

    parsed: list[np.datetime64] = []
    for item in values:
        dt = _parse_date_like(item)
        parsed.append(np.datetime64(dt, "D") if dt is not None else np.datetime64("NaT", "D"))
    return np.array(parsed, dtype="datetime64[D]")


def _to_float_array(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray):
        if np.issubdtype(values.dtype, np.number):
            return values.astype(np.float64, copy=False)
        values_iter: Sequence[Any] = values.tolist()
    else:
        values_iter = values

    out = np.full(len(values_iter), np.nan, dtype=np.float64)
    for idx, item in enumerate(values_iter):
        parsed = _coerce_float(item)
        if parsed is not None:
            out[idx] = parsed
    return out


def _broadcast_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if left.shape == right.shape:
        return left, right
    if left.size == 1:
        return np.broadcast_to(left, right.shape).astype(np.float64), right
    if right.size == 1:
        return left, np.broadcast_to(right, left.shape).astype(np.float64)
    raise ValueError("Input arrays must have same shape or be broadcastable scalars.")


def _to_optional_list(values: np.ndarray, *, round_digits: int) -> list[float | None]:
    out: list[float | None] = []
    for item in values:
        if not np.isfinite(item):
            out.append(None)
        else:
            out.append(round(float(item), round_digits))
    return out


def _datetime64_to_iso_list(values: np.ndarray) -> list[str | None]:
    out: list[str | None] = []
    for item in values.astype("datetime64[D]"):
        if np.isnat(item):
            out.append(None)
        else:
            out.append(str(item))
    return out


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return float(parsed) if np.isfinite(parsed) else None


def _parse_date_like(value: Any) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        text = str(value.astype("datetime64[D]"))
        return date.fromisoformat(text)
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
