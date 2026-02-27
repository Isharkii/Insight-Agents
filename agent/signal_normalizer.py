"""
agent/signal_normalizer.py

Normalize nested KPI and forecast payloads into a strict flat signal contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import logging

logger = logging.getLogger(__name__)

SignalDict = dict[str, float]

REQUIRED_SIGNALS = [
    "revenue_growth_delta",
    "churn_delta",
    "conversion_delta",
    "slope",
    "deviation_percentage",
]


def normalize_signals(
    kpi_payload: dict,
    forecast_payload: dict,
) -> SignalDict:
    """
    Normalize KPI and forecast payloads into canonical flat signals.

    Returned keys:
        - revenue_growth_delta
        - churn_delta
        - conversion_delta
        - slope
        - deviation_percentage
        - churn_acceleration

    Raises
    ------
    ValueError
        If any required signal cannot be extracted or derived.
    """
    kpi_signals = normalize_kpi_signals(kpi_payload)
    forecast_signals = normalize_forecast_signals(forecast_payload)

    signals: SignalDict = {
        **kpi_signals,
        **forecast_signals,
    }
    _validate_required_signals(signals)
    return signals


def normalize_kpi_signals(
    kpi_payload: dict,
    *,
    strict: bool = False,
) -> SignalDict:
    """Normalize only KPI-derived flat signals.

    When *strict* is False (default), derivation failures for individual
    signals fall back to 0.0 and are recorded as warnings.  This allows
    downstream consumers (risk_node) to degrade gracefully instead of
    hard-failing the entire pipeline.

    When *strict* is True the original behaviour is preserved: any
    derivation failure raises ``ValueError``.

    The returned dict includes a ``_warnings`` key (list[str]) that
    callers should inspect and propagate.
    """
    kpi_series = _extract_kpi_series(kpi_payload)
    warnings: list[str] = []
    signals: SignalDict = {}

    for name, derive_fn in (
        ("revenue_growth_delta", _derive_revenue_growth_delta),
        ("churn_delta", _derive_churn_delta),
        ("conversion_delta", _derive_conversion_delta),
    ):
        try:
            signals[name] = derive_fn(kpi_series)
        except ValueError as exc:
            if strict:
                raise
            signals[name] = 0.0
            warnings.append(f"{name}: defaulted to 0.0 ({exc})")
            logger.warning("KPI signal '%s' derivation failed, defaulting to 0.0: %s", name, exc)

    signals["_warnings"] = warnings  # type: ignore[assignment]
    return signals


@dataclass
class KPIReadinessResult:
    """Pre-flight assessment of KPI payload readiness for risk scoring."""

    is_ready: bool
    """True if there is enough data to attempt signal derivation."""

    record_count: int
    """Number of KPI records with usable computed_kpis."""

    available_metrics: list[str]
    """Metric names found across all records."""

    missing_revenue_proxy: bool
    """True if no revenue-family metric is available."""

    missing_churn_proxy: bool
    """True if no churn-family metric is available."""

    missing_conversion_proxy: bool
    """True if no conversion-family metric is available."""

    max_series_depth: int
    """Deepest time-series across all metrics (number of periods)."""

    reasons: list[str]
    """Human-readable list of reasons the data is not ready."""


_REVENUE_FAMILY = frozenset({
    "revenue_growth_delta", "growth_rate", "revenue", "mrr",
    "total_revenue", "retainer_revenue",
})
_CHURN_FAMILY = frozenset({
    "churn_delta", "churn_rate", "client_churn",
})
_CONVERSION_FAMILY = frozenset({
    "conversion_delta", "conversion_rate",
})

# Minimum time-series depth needed for delta/pct-change derivation.
MIN_SERIES_DEPTH_FOR_DELTA = 2


def check_kpi_readiness(kpi_payload: dict) -> KPIReadinessResult:
    """Non-throwing pre-flight check on KPI payload data readiness.

    Inspects the payload structure and metric availability without
    attempting signal derivation.  Returns a structured result that
    callers (risk_node) can use to decide between ``success``,
    ``insufficient_data``, or ``failed``.
    """
    reasons: list[str] = []

    if not isinstance(kpi_payload, dict):
        return KPIReadinessResult(
            is_ready=False, record_count=0, available_metrics=[],
            missing_revenue_proxy=True, missing_churn_proxy=True,
            missing_conversion_proxy=True, max_series_depth=0,
            reasons=["kpi_payload is not a dict"],
        )

    records = kpi_payload.get("records")
    if not isinstance(records, list) or not records:
        return KPIReadinessResult(
            is_ready=False, record_count=0, available_metrics=[],
            missing_revenue_proxy=True, missing_churn_proxy=True,
            missing_conversion_proxy=True, max_series_depth=0,
            reasons=["kpi_payload.records is empty or missing"],
        )

    # Count usable records and build metric inventory.
    metric_series_depth: dict[str, int] = {}
    usable_record_count = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        computed = record.get("computed_kpis")
        if not isinstance(computed, dict):
            continue
        usable_record_count += 1
        for metric_name, raw_value in computed.items():
            try:
                _metric_to_float(raw_value)
            except ValueError:
                continue
            metric_series_depth[metric_name] = metric_series_depth.get(metric_name, 0) + 1

    if usable_record_count == 0:
        return KPIReadinessResult(
            is_ready=False, record_count=0, available_metrics=[],
            missing_revenue_proxy=True, missing_churn_proxy=True,
            missing_conversion_proxy=True, max_series_depth=0,
            reasons=["no records contain usable computed_kpis"],
        )

    available = set(metric_series_depth.keys())
    max_depth = max(metric_series_depth.values()) if metric_series_depth else 0

    missing_rev = not bool(available & _REVENUE_FAMILY)
    missing_churn = not bool(available & _CHURN_FAMILY)
    missing_conv = not bool(available & _CONVERSION_FAMILY)

    if missing_rev:
        reasons.append(
            f"no revenue-family metric found (need one of: {sorted(_REVENUE_FAMILY)})"
        )
    if missing_churn:
        reasons.append(
            f"no churn-family metric found (need one of: {sorted(_CHURN_FAMILY)})"
        )
    if missing_conv:
        reasons.append(
            f"no conversion-family metric found (need one of: {sorted(_CONVERSION_FAMILY)})"
        )
    if max_depth < MIN_SERIES_DEPTH_FOR_DELTA:
        reasons.append(
            f"max time-series depth is {max_depth}, need >= {MIN_SERIES_DEPTH_FOR_DELTA} for delta derivation"
        )

    is_ready = not reasons
    return KPIReadinessResult(
        is_ready=is_ready,
        record_count=usable_record_count,
        available_metrics=sorted(available),
        missing_revenue_proxy=missing_rev,
        missing_churn_proxy=missing_churn,
        missing_conversion_proxy=missing_conv,
        max_series_depth=max_depth,
        reasons=reasons,
    )


def normalize_forecast_signals(forecast_payload: dict) -> SignalDict:
    """Normalize only forecast-derived flat signals."""
    forecast_rows = _extract_forecast_rows(forecast_payload)
    return {
        "slope": _extract_forecast_signal(forecast_rows, "slope"),
        "deviation_percentage": _extract_forecast_signal(
            forecast_rows,
            "deviation_percentage",
        ),
        "churn_acceleration": _derive_churn_acceleration(forecast_rows),
    }


def _validate_required_signals(signals: dict[str, Any]) -> None:
    missing: list[str] = []
    invalid: list[str] = []

    for key in REQUIRED_SIGNALS:
        if key not in signals:
            missing.append(key)
            continue
        value = signals[key]
        if value is None:
            invalid.append(f"{key}=None")
            continue
        try:
            float(value)
        except (TypeError, ValueError):
            invalid.append(f"{key}={value!r}")

    if missing or invalid:
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if invalid:
            details.append(f"invalid: {', '.join(invalid)}")
        raise ValueError(
            "Signal normalization failed required-signal validation "
            f"({'; '.join(details)})."
        )


def _extract_kpi_series(kpi_payload: dict) -> dict[str, list[float]]:
    if not isinstance(kpi_payload, dict):
        raise ValueError("kpi_payload must be a dict.")

    records = kpi_payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("kpi_payload.records must be a non-empty list.")

    ordered_records = sorted(records, key=_record_sort_key)
    series: dict[str, list[float]] = {}

    for idx, record in enumerate(ordered_records):
        if not isinstance(record, dict):
            raise ValueError(f"kpi_payload.records[{idx}] must be a dict.")

        computed = record.get("computed_kpis")
        if not isinstance(computed, dict):
            continue

        for metric_name, raw_value in computed.items():
            try:
                numeric_value = _metric_to_float(raw_value)
            except ValueError:
                continue
            series.setdefault(metric_name, []).append(numeric_value)

    if not series:
        raise ValueError("No usable computed_kpis values found in kpi_payload.")

    return series


def _extract_forecast_rows(forecast_payload: dict) -> list[dict[str, Any]]:
    if not isinstance(forecast_payload, dict):
        raise ValueError("forecast_payload must be a dict.")

    forecasts = forecast_payload.get("forecasts")
    if not isinstance(forecasts, dict) or not forecasts:
        raise ValueError("forecast_payload.forecasts must be a non-empty dict.")

    rows: list[dict[str, Any]] = []
    for metric_key, row in forecasts.items():
        if row is None:
            continue
        if not isinstance(row, dict):
            raise ValueError(f"forecast row for metric '{metric_key}' must be a dict or None.")
        payload = row.get("forecast_data")
        if isinstance(payload, dict):
            rows.append({"metric": str(metric_key), "payload": payload})

    if not rows:
        raise ValueError("No usable forecast_data rows found in forecast_payload.forecasts.")

    return rows


def _metric_to_float(raw_value: Any) -> float:
    if isinstance(raw_value, dict):
        if raw_value.get("error") is not None:
            raise ValueError("metric has error.")
        raw_value = raw_value.get("value")
    if raw_value is None:
        raise ValueError("metric value is None.")
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("metric value is not numeric.") from exc


def _record_sort_key(record: Any) -> tuple[datetime, datetime]:
    if not isinstance(record, dict):
        return (datetime.min, datetime.min)
    return (
        _parse_iso_datetime(record.get("period_end")),
        _parse_iso_datetime(record.get("created_at")),
    )


def _parse_iso_datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        return datetime.min
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.min


def _latest(series: dict[str, list[float]], key: str) -> float | None:
    values = series.get(key)
    if values:
        return values[-1]
    return None


def _delta_from_series(series: dict[str, list[float]], key: str) -> float | None:
    values = series.get(key)
    if values and len(values) >= 2:
        return values[-1] - values[-2]
    return None


def _pct_change_from_series(series: dict[str, list[float]], key: str) -> float | None:
    values = series.get(key)
    if not values or len(values) < 2:
        return None
    prev = values[-2]
    curr = values[-1]
    if prev == 0.0:
        raise ValueError(
            f"Cannot derive revenue_growth_delta from '{key}' because previous value is 0."
        )
    return (curr - prev) / abs(prev)


def _first_available(*candidates: float | None) -> float | None:
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def _require(name: str, value: float | None) -> float:
    if value is None:
        raise ValueError(f"Unable to derive required signal '{name}'.")
    return float(value)


def _derive_revenue_growth_delta(series: dict[str, list[float]]) -> float:
    direct = _latest(series, "revenue_growth_delta")
    from_growth_rate = _latest(series, "growth_rate")
    from_revenue = _pct_change_from_series(series, "revenue")
    from_mrr = _pct_change_from_series(series, "mrr")
    from_total_revenue = _pct_change_from_series(series, "total_revenue")
    from_retainer = _pct_change_from_series(series, "retainer_revenue")

    return _require(
        "revenue_growth_delta",
        _first_available(
            direct,
            from_growth_rate,
            from_revenue,
            from_mrr,
            from_total_revenue,
            from_retainer,
        ),
    )


def _derive_churn_delta(series: dict[str, list[float]]) -> float:
    direct = _latest(series, "churn_delta")
    churn_rate_delta = _delta_from_series(series, "churn_rate")
    client_churn_delta = _delta_from_series(series, "client_churn")
    churn_rate_level = _latest(series, "churn_rate")
    client_churn_level = _latest(series, "client_churn")

    return _require(
        "churn_delta",
        _first_available(
            direct,
            churn_rate_delta,
            client_churn_delta,
            churn_rate_level,
            client_churn_level,
        ),
    )


def _derive_conversion_delta(series: dict[str, list[float]]) -> float:
    direct = _latest(series, "conversion_delta")
    from_conversion_rate_delta = _delta_from_series(series, "conversion_rate")
    from_conversion_rate_level = _latest(series, "conversion_rate")

    return _require(
        "conversion_delta",
        _first_available(
            direct,
            from_conversion_rate_delta,
            from_conversion_rate_level,
        ),
    )


def _extract_forecast_signal(rows: list[dict[str, Any]], key: str) -> float:
    for row in rows:
        payload = row["payload"]
        if key in payload and payload[key] is not None:
            try:
                return float(payload[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Forecast signal '{key}' is not numeric.") from exc
    raise ValueError(f"Unable to derive required forecast signal '{key}'.")


def _derive_churn_acceleration(rows: list[dict[str, Any]]) -> float:
    direct = _extract_optional_forecast_signal(rows, "churn_acceleration")
    if direct is not None:
        return direct

    churn_row = _find_churn_forecast_row(rows)
    if churn_row is None:
        raise ValueError("Unable to derive required signal 'churn_acceleration'.")

    payload = churn_row["payload"]
    forecast = payload.get("forecast")
    if isinstance(forecast, dict):
        m1 = _coerce_optional_float(forecast.get("month_1"))
        m2 = _coerce_optional_float(forecast.get("month_2"))
        m3 = _coerce_optional_float(forecast.get("month_3"))
        if m1 is not None and m2 is not None and m3 is not None:
            return m3 - (2.0 * m2) + m1

    slope = _coerce_optional_float(payload.get("slope"))
    if slope is not None:
        return slope

    raise ValueError("Unable to derive required signal 'churn_acceleration'.")


def _extract_optional_forecast_signal(
    rows: list[dict[str, Any]],
    key: str,
) -> float | None:
    for row in rows:
        payload = row["payload"]
        if key in payload and payload[key] is not None:
            try:
                return float(payload[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Forecast signal '{key}' is not numeric.") from exc
    return None


def _find_churn_forecast_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        metric = row["metric"].lower()
        payload = row["payload"]
        metric_name = str(payload.get("metric_name", "")).lower()
        if "churn" in metric or "churn" in metric_name:
            return row
    return None


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
