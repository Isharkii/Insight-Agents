"""
agent/signal_normalizer.py

Normalize nested KPI and forecast payloads into a strict flat signal contract.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


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
    kpi_series = _extract_kpi_series(kpi_payload)
    forecast_rows = _extract_forecast_rows(forecast_payload)

    revenue_growth_delta = _derive_revenue_growth_delta(kpi_series)
    churn_delta = _derive_churn_delta(kpi_series)
    conversion_delta = _derive_conversion_delta(kpi_series)

    slope = _extract_forecast_signal(forecast_rows, "slope")
    deviation_percentage = _extract_forecast_signal(
        forecast_rows,
        "deviation_percentage",
    )
    signals: SignalDict = {
        "revenue_growth_delta": revenue_growth_delta,
        "churn_delta": churn_delta,
        "conversion_delta": conversion_delta,
        "slope": slope,
        "deviation_percentage": deviation_percentage,
    }
    _validate_required_signals(signals)

    signals["churn_acceleration"] = _derive_churn_acceleration(forecast_rows)

    return signals


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
