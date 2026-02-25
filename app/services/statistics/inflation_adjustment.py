from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, Mapping, Sequence

import numpy as np


def align_cpi_to_kpi_periods(
    kpi_period_ends: Sequence[Any],
    cpi_rows: Sequence[Mapping[str, Any]],
    *,
    cpi_key: str = "cpi_index",
) -> np.ndarray:
    """
    Align CPI values to KPI period-end dates using interval containment.

    A CPI row matches a KPI period when:
    period_start <= kpi_period_end <= period_end

    This supports monthly and quarterly CPI windows without exact-date
    matching assumptions.
    """

    periods = _to_datetime64_day(kpi_period_ends)
    if len(cpi_rows) == 0:
        return np.full(periods.shape, np.nan, dtype=np.float64)

    starts, ends, values = _extract_interval_values(cpi_rows, value_key=cpi_key)
    if starts.size == 0:
        return np.full(periods.shape, np.nan, dtype=np.float64)

    order = np.argsort(starts.astype(np.int64), kind="stable")
    starts = starts[order]
    ends = ends[order]
    values = values[order]

    aligned = np.full(periods.shape, np.nan, dtype=np.float64)
    valid_periods = ~np.isnat(periods)
    if not np.any(valid_periods):
        return aligned

    valid_dates = periods[valid_periods]
    idx = np.searchsorted(starts, valid_dates, side="right") - 1
    has_candidate = idx >= 0
    if not np.any(has_candidate):
        return aligned

    candidate_idx = idx[has_candidate]
    candidate_dates = valid_dates[has_candidate]
    within = candidate_dates <= ends[candidate_idx]
    target_positions = np.flatnonzero(valid_periods)[has_candidate][within]
    aligned[target_positions] = values[candidate_idx[within]]
    return aligned


def compute_cpi_inflation_rate(
    cpi_index: Sequence[Any] | np.ndarray,
    *,
    missing_policy: Literal["nan", "ffill"] = "ffill",
) -> np.ndarray:
    """
    Convert CPI index levels into period-over-period inflation rates.

    Formula:
    inflation_t = (CPI_t / CPI_{t-1}) - 1

    Missing CPI is handled by policy:
    - ``nan``: keep missing periods as NaN
    - ``ffill``: carry prior CPI index forward where available
    """

    index = _to_float_array(cpi_index)
    filled = _apply_missing_policy(index, policy=missing_policy)

    out = np.full(filled.shape, np.nan, dtype=np.float64)
    if filled.size <= 1:
        return out

    previous = filled[:-1]
    current = filled[1:]
    valid = np.isfinite(previous) & np.isfinite(current) & (np.abs(previous) > 1e-12)
    rates = np.full(previous.shape, np.nan, dtype=np.float64)
    rates[valid] = (current[valid] / previous[valid]) - 1.0
    out[1:] = rates
    return out


def compute_real_growth_rate(
    nominal_growth_rate: Sequence[Any] | np.ndarray,
    inflation_rate: Sequence[Any] | np.ndarray,
    *,
    normalize_percent: bool = True,
) -> np.ndarray:
    """
    Compute inflation-adjusted real growth from nominal growth.

    Formula:
    real_growth = ((1 + nominal_growth) / (1 + inflation_rate)) - 1

    Economic reasoning:
    Nominal growth includes general price increases. Deflating by CPI-based
    inflation isolates growth in real purchasing-power terms.
    """

    nominal = _to_float_array(nominal_growth_rate)
    inflation = _to_float_array(inflation_rate)
    nominal, inflation = _broadcast_pair(nominal, inflation)
    if normalize_percent:
        nominal = _normalize_rate_array(nominal)
        inflation = _normalize_rate_array(inflation)

    denominator = 1.0 + inflation
    out = np.full(nominal.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(nominal) & np.isfinite(inflation) & (np.abs(denominator) > 1e-12)
    out[valid] = ((1.0 + nominal[valid]) / denominator[valid]) - 1.0
    return out


def compute_inflation_adjusted_revenue(
    revenue: Sequence[Any] | np.ndarray,
    cpi_index: Sequence[Any] | np.ndarray,
    *,
    base_cpi: float | None = None,
    missing_policy: Literal["nan", "ffill"] = "ffill",
) -> np.ndarray:
    """
    Deflate nominal revenue into real revenue by CPI index.

    Formula:
    inflation_adjusted_revenue = revenue / (CPI_index / base_CPI)

    If ``base_cpi`` is not provided, the first available CPI value is used.
    Missing CPI data is handled by the requested policy and returns NaN when
    no valid CPI baseline is available.
    """

    revenue_arr = _to_float_array(revenue)
    cpi_arr = _to_float_array(cpi_index)
    revenue_arr, cpi_arr = _broadcast_pair(revenue_arr, cpi_arr)
    cpi_arr = _apply_missing_policy(cpi_arr, policy=missing_policy)

    base = float(base_cpi) if base_cpi is not None else _first_finite(cpi_arr)
    if base is None or not np.isfinite(base) or abs(base) <= 1e-12:
        return np.full(revenue_arr.shape, np.nan, dtype=np.float64)

    scale = cpi_arr / base
    out = np.full(revenue_arr.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(revenue_arr) & np.isfinite(scale) & (np.abs(scale) > 1e-12)
    out[valid] = revenue_arr[valid] / scale[valid]
    return out


def compound_inflation_rates(
    inflation_rate: Sequence[Any] | np.ndarray,
    *,
    normalize_percent: bool = True,
    missing_policy: Literal["nan", "ffill"] = "ffill",
) -> np.ndarray:
    """
    Compute cumulative multi-period inflation from per-period inflation rates.

    Formula:
    cumulative_inflation_t = prod_{i=1..t}(1 + inflation_i) - 1

    Missing inflation observations are handled by policy to avoid pipeline
    failure on partial macro data.
    """

    rates = _to_float_array(inflation_rate)
    if normalize_percent:
        rates = _normalize_rate_array(rates)
    rates = _apply_missing_policy(rates, policy=missing_policy)

    out = np.full(rates.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(rates)
    if not np.any(finite):
        return out

    first_valid = int(np.flatnonzero(finite)[0])
    working = rates[first_valid:].copy()

    # For policy="nan", preserve NaN gaps by isolating contiguous finite blocks.
    if missing_policy == "nan":
        starts = np.flatnonzero(np.isfinite(working) & np.concatenate(([True], ~np.isfinite(working[:-1]))))
        for start in starts:
            tail = working[start:]
            finite_tail = np.isfinite(tail)
            run_len = int(np.argmax(~finite_tail)) if np.any(~finite_tail) else len(tail)
            if run_len == 0:
                continue
            run = tail[:run_len]
            compounded = np.cumprod(1.0 + run) - 1.0
            out[first_valid + start : first_valid + start + run_len] = compounded
        return out

    compounded = np.cumprod(1.0 + working) - 1.0
    out[first_valid:] = compounded
    return out


def build_inflation_adjusted_series(
    *,
    kpi_period_ends: Sequence[Any],
    nominal_growth_rate: Sequence[Any],
    revenue: Sequence[Any],
    cpi_rows: Sequence[Mapping[str, Any]],
    cpi_key: str = "cpi_index",
    base_cpi: float | None = None,
    missing_policy: Literal["nan", "ffill"] = "ffill",
) -> dict[str, Any]:
    """
    End-to-end helper to align CPI and compute inflation-adjusted outputs.
    """

    periods = _to_datetime64_day(kpi_period_ends)
    cpi_index = align_cpi_to_kpi_periods(periods, cpi_rows, cpi_key=cpi_key)
    cpi_inflation = compute_cpi_inflation_rate(cpi_index, missing_policy=missing_policy)
    real_growth = compute_real_growth_rate(nominal_growth_rate, cpi_inflation)
    adjusted_revenue = compute_inflation_adjusted_revenue(
        revenue,
        cpi_index,
        base_cpi=base_cpi,
        missing_policy=missing_policy,
    )
    compounded = compound_inflation_rates(cpi_inflation, normalize_percent=False, missing_policy=missing_policy)

    return {
        "formulas": {
            "real_growth_rate": "((1 + nominal_growth) / (1 + inflation_rate)) - 1",
            "inflation_adjusted_revenue": "revenue / (CPI_index / base_CPI)",
            "multi_period_inflation_compounding": "prod(1 + inflation_rate) - 1",
        },
        "period_end": _datetime64_to_list(periods),
        "cpi_index": _to_optional_list(cpi_index),
        "cpi_inflation_rate": _to_optional_list(cpi_inflation),
        "real_growth_rate": _to_optional_list(real_growth),
        "inflation_adjusted_revenue": _to_optional_list(adjusted_revenue),
        "multi_period_inflation": _to_optional_list(compounded),
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
        value_raw = row.get(value_key, row.get("value"))

        start_date = _parse_date_like(start_raw)
        end_date = _parse_date_like(end_raw)
        value = _coerce_float(value_raw)

        if start_date is None or end_date is None or value is None:
            continue

        starts.append(np.datetime64(start_date, "D"))
        ends.append(np.datetime64(end_date, "D"))
        values.append(value)

    if not starts:
        empty_dates = np.array([], dtype="datetime64[D]")
        return empty_dates, empty_dates, np.array([], dtype=np.float64)

    return (
        np.array(starts, dtype="datetime64[D]"),
        np.array(ends, dtype="datetime64[D]"),
        np.array(values, dtype=np.float64),
    )


def _to_datetime64_day(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[D]")
    out: list[np.datetime64] = []
    for value in values:
        parsed = _parse_date_like(value)
        out.append(np.datetime64(parsed, "D") if parsed is not None else np.datetime64("NaT", "D"))
    return np.array(out, dtype="datetime64[D]")


def _to_float_array(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray):
        if np.issubdtype(values.dtype, np.number):
            return values.astype(np.float64, copy=False)
        sequence: Sequence[Any] = values.tolist()
    else:
        sequence = values

    out = np.full(len(sequence), np.nan, dtype=np.float64)
    for idx, value in enumerate(sequence):
        parsed = _coerce_float(value)
        if parsed is not None:
            out[idx] = parsed
    return out


def _normalize_rate_array(rates: np.ndarray, *, threshold: float = 1.0, ceiling: float = 500.0) -> np.ndarray:
    out = rates.astype(np.float64, copy=True)
    finite = np.isfinite(out)
    percent_like = finite & (np.abs(out) >= threshold) & (np.abs(out) <= ceiling)
    out[percent_like] = out[percent_like] / 100.0
    return out


def _apply_missing_policy(values: np.ndarray, *, policy: Literal["nan", "ffill"]) -> np.ndarray:
    if policy == "nan":
        return values.astype(np.float64, copy=True)
    if policy != "ffill":
        raise ValueError(f"Unsupported missing policy: {policy}")
    return _forward_fill(values)


def _forward_fill(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float64, copy=True)
    finite = np.isfinite(out)
    if not np.any(finite):
        return out

    idx = np.where(finite, np.arange(out.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = out[idx]

    first_valid = int(np.flatnonzero(finite)[0])
    if first_valid > 0:
        out[:first_valid] = np.nan
    return out


def _broadcast_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if left.shape == right.shape:
        return left, right
    if left.size == 1:
        return np.broadcast_to(left, right.shape).astype(np.float64), right
    if right.size == 1:
        return left, np.broadcast_to(right, left.shape).astype(np.float64)
    raise ValueError("Input arrays must have same shape or be broadcastable scalars.")


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


def _first_finite(values: np.ndarray) -> float | None:
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return None
    return float(values[int(finite[0])])


def _to_optional_list(values: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for value in values:
        if np.isfinite(value):
            out.append(round(float(value), 6))
        else:
            out.append(None)
    return out


def _datetime64_to_list(values: np.ndarray) -> list[str | None]:
    out: list[str | None] = []
    for value in values.astype("datetime64[D]"):
        if np.isnat(value):
            out.append(None)
        else:
            out.append(str(value))
    return out

