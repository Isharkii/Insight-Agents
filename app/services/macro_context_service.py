from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


INFLATION_METRIC_HINTS: tuple[str, ...] = (
    "cpi",
    "inflation",
    "consumer_price",
    "consumer_price_index",
)

BENCHMARK_METRIC_HINTS: tuple[str, ...] = (
    "industry_benchmark",
    "benchmark",
    "peer_index",
    "industry_index",
    "industry_growth",
)


def build_macro_context(
    *,
    kpi_payload: Mapping[str, Any],
    inflation_rows: Sequence[Mapping[str, Any]] | None = None,
    benchmark_rows: Sequence[Mapping[str, Any]] | None = None,
    metric_candidates: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Build deterministic macro context for insight payloads.

    CPI/inflation and benchmark series are optional. Missing external inputs
    are marked explicitly with status flags so the pipeline remains runnable.
    """
    entity_series = _extract_entity_series(
        kpi_payload=kpi_payload,
        metric_candidates=metric_candidates or (),
    )

    seasonality = _detect_seasonality(entity_series)
    inflation = _build_external_signal(
        rows=inflation_rows or (),
        metric_filter=is_inflation_metric_name,
        signal_name="inflation",
    )
    benchmark = _build_external_signal(
        rows=benchmark_rows or (),
        metric_filter=is_benchmark_metric_name,
        signal_name="industry_benchmark",
    )

    nominal_growth_rate = _series_growth_rate(entity_series)
    inflation_rate = inflation.get("derived_rate")
    real_growth = _real_growth_adjustment(nominal_growth_rate, inflation_rate)
    benchmark_gap = _benchmark_gap(nominal_growth_rate, real_growth, benchmark.get("derived_rate"))

    has_optional_gaps = any(
        item.get("status") != "available"
        for item in (inflation, benchmark)
    )
    overall_status = "partial" if has_optional_gaps else "success"

    return {
        "status": overall_status,
        "signals": {
            "seasonality": seasonality,
            "inflation": inflation,
            "industry_benchmark": benchmark,
        },
        "real_growth_adjustment": real_growth,
        "benchmark_comparison": benchmark_gap,
        "reproducibility": {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "entity_series_points": len(entity_series),
            "entity_series_start": entity_series[0]["timestamp"] if entity_series else None,
            "entity_series_end": entity_series[-1]["timestamp"] if entity_series else None,
            "benchmark_sources": benchmark.get("source_observations", []),
            "inflation_sources": inflation.get("source_observations", []),
        },
    }


def is_inflation_metric_name(metric_name: str | None) -> bool:
    normalized = _normalize(metric_name)
    return bool(normalized) and any(hint in normalized for hint in INFLATION_METRIC_HINTS)


def is_benchmark_metric_name(metric_name: str | None) -> bool:
    normalized = _normalize(metric_name)
    return bool(normalized) and any(hint in normalized for hint in BENCHMARK_METRIC_HINTS)


def _extract_entity_series(
    *,
    kpi_payload: Mapping[str, Any],
    metric_candidates: Sequence[str],
) -> list[dict[str, Any]]:
    records = kpi_payload.get("records")
    if not isinstance(records, list):
        return []

    normalized_candidates = {_normalize(name) for name in metric_candidates if _normalize(name)}
    by_metric: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        if not isinstance(record, Mapping):
            continue
        ts = _parse_iso_datetime(record.get("period_end")) or _parse_iso_datetime(record.get("created_at"))
        if ts is None:
            continue

        computed = record.get("computed_kpis")
        if not isinstance(computed, Mapping):
            continue

        for raw_metric_name, raw_metric_value in computed.items():
            metric_name = str(raw_metric_name).strip()
            normalized = _normalize(metric_name)
            if normalized_candidates and normalized not in normalized_candidates:
                continue
            numeric = _coerce_float(raw_metric_value)
            if numeric is None:
                continue
            bucket = by_metric.setdefault(metric_name, [])
            bucket.append(
                {
                    "timestamp": ts.isoformat(),
                    "value": numeric,
                    "metric_name": metric_name,
                    "source_type": "computed_kpi",
                    "metadata": {
                        "record_created_at": record.get("created_at"),
                    },
                }
            )

    if not by_metric:
        return []

    selected_metric = max(
        by_metric.items(),
        key=lambda item: (len(item[1]), item[0]),
    )[0]
    points = by_metric[selected_metric]
    points.sort(key=lambda item: (item["timestamp"], str(item["metric_name"])))
    return points


def _detect_seasonality(series: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(series) < 12:
        return {
            "status": "insufficient_history",
            "is_seasonal": False,
            "seasonality_strength": 0.0,
            "points_used": len(series),
            "method": "month_of_year_variance_ratio",
        }

    monthly_values: dict[int, list[float]] = {}
    values: list[float] = []
    for point in series:
        ts = _parse_iso_datetime(point.get("timestamp"))
        value = _coerce_float(point.get("value"))
        if ts is None or value is None:
            continue
        values.append(value)
        monthly_values.setdefault(ts.month, []).append(value)

    if len(values) < 12:
        return {
            "status": "insufficient_history",
            "is_seasonal": False,
            "seasonality_strength": 0.0,
            "points_used": len(values),
            "method": "month_of_year_variance_ratio",
        }

    total_variance = _variance(values)
    if total_variance <= 0.0:
        return {
            "status": "available",
            "is_seasonal": False,
            "seasonality_strength": 0.0,
            "points_used": len(values),
            "coverage_months": len(monthly_values),
            "method": "month_of_year_variance_ratio",
        }

    month_means = [_mean(bucket) for bucket in monthly_values.values() if bucket]
    seasonal_variance = _variance(month_means)
    strength = max(0.0, min(1.0, seasonal_variance / total_variance))
    is_seasonal = strength >= 0.15 and len(monthly_values) >= 6

    peak_month, trough_month = _seasonal_extremes(monthly_values)
    return {
        "status": "available",
        "is_seasonal": is_seasonal,
        "seasonality_strength": round(strength, 6),
        "points_used": len(values),
        "coverage_months": len(monthly_values),
        "peak_month": peak_month,
        "trough_month": trough_month,
        "method": "month_of_year_variance_ratio",
    }


def _build_external_signal(
    *,
    rows: Sequence[Mapping[str, Any]],
    metric_filter: Any,
    signal_name: str,
) -> dict[str, Any]:
    points = _normalize_external_rows(rows, metric_filter=metric_filter)
    if not points:
        return {
            "status": "missing_optional",
            "signal": signal_name,
            "points_used": 0,
            "latest_value": None,
            "latest_timestamp": None,
            "derived_rate": None,
            "source_observations": [],
        }

    latest = points[-1]
    derived_rate = _derive_rate(points, metric_name=str(latest.get("metric_name") or ""))
    return {
        "status": "available",
        "signal": signal_name,
        "points_used": len(points),
        "latest_value": latest["value"],
        "latest_timestamp": latest["timestamp"],
        "derived_rate": derived_rate,
        "source_observations": points,
    }


def _normalize_external_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_filter: Any,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        metric_name = str(row.get("metric_name") or "").strip()
        if not metric_filter(metric_name):
            continue
        timestamp = _parse_iso_datetime(row.get("timestamp"))
        numeric = _coerce_float(row.get("metric_value"))
        if timestamp is None or numeric is None:
            continue
        metadata = row.get("metadata_json")
        if not isinstance(metadata, Mapping):
            metadata = {}
        out.append(
            {
                "timestamp": timestamp.isoformat(),
                "value": numeric,
                "metric_name": metric_name,
                "source_type": str(row.get("source_type") or "").strip().lower() or None,
                "entity_name": str(row.get("entity_name") or "").strip() or None,
                "category": str(row.get("category") or "").strip() or None,
                "metadata": dict(metadata),
            }
        )
    out.sort(key=lambda item: (item["timestamp"], str(item["metric_name"])))
    return out


def _series_growth_rate(series: Sequence[Mapping[str, Any]]) -> float | None:
    if len(series) < 2:
        return None
    first = _coerce_float(series[0].get("value"))
    last = _coerce_float(series[-1].get("value"))
    if first is None or last is None or first == 0.0:
        return None
    return round((last - first) / abs(first), 6)


def _derive_rate(points: Sequence[Mapping[str, Any]], *, metric_name: str) -> float | None:
    if not points:
        return None

    normalized_name = _normalize(metric_name)
    if "cpi" in normalized_name and "inflation" not in normalized_name:
        rate = _series_growth_rate(points)
        if rate is not None:
            return rate

    if len(points) >= 2:
        growth = _series_growth_rate(points)
        if growth is not None:
            return growth

    latest = _coerce_float(points[-1].get("value"))
    return _normalize_rate(latest)


def _real_growth_adjustment(
    nominal_growth_rate: float | None,
    inflation_rate: float | None,
) -> dict[str, Any]:
    if nominal_growth_rate is None or inflation_rate is None:
        return {
            "status": "missing_optional",
            "nominal_growth_rate": nominal_growth_rate,
            "inflation_rate": inflation_rate,
            "real_growth_rate": None,
            "formula": "((1 + nominal_growth_rate) / (1 + inflation_rate)) - 1",
        }

    denominator = 1.0 + inflation_rate
    if denominator == 0.0:
        return {
            "status": "missing_optional",
            "nominal_growth_rate": nominal_growth_rate,
            "inflation_rate": inflation_rate,
            "real_growth_rate": None,
            "formula": "((1 + nominal_growth_rate) / (1 + inflation_rate)) - 1",
        }

    real_growth = ((1.0 + nominal_growth_rate) / denominator) - 1.0
    return {
        "status": "available",
        "nominal_growth_rate": nominal_growth_rate,
        "inflation_rate": inflation_rate,
        "real_growth_rate": round(real_growth, 6),
        "formula": "((1 + nominal_growth_rate) / (1 + inflation_rate)) - 1",
    }


def _benchmark_gap(
    nominal_growth_rate: float | None,
    real_growth_adjustment: Mapping[str, Any],
    benchmark_rate: float | None,
) -> dict[str, Any]:
    real_growth = _coerce_float(real_growth_adjustment.get("real_growth_rate"))
    if nominal_growth_rate is None or benchmark_rate is None:
        return {
            "status": "missing_optional",
            "benchmark_growth_rate": benchmark_rate,
            "nominal_growth_gap": None,
            "real_growth_gap": None,
        }

    nominal_gap = nominal_growth_rate - benchmark_rate
    real_gap = (real_growth - benchmark_rate) if real_growth is not None else None
    return {
        "status": "available",
        "benchmark_growth_rate": benchmark_rate,
        "nominal_growth_gap": round(nominal_gap, 6),
        "real_growth_gap": round(real_gap, 6) if real_gap is not None else None,
    }


def _seasonal_extremes(monthly_values: Mapping[int, Sequence[float]]) -> tuple[int | None, int | None]:
    if not monthly_values:
        return None, None
    averages = {
        month: _mean([value for value in values if isinstance(value, (int, float))])
        for month, values in monthly_values.items()
        if values
    }
    if not averages:
        return None, None
    peak = max(averages.items(), key=lambda item: (item[1], item[0]))[0]
    trough = min(averages.items(), key=lambda item: (item[1], item[0]))[0]
    return int(peak), int(trough)


def _mean(values: Sequence[float]) -> float:
    filtered = [float(value) for value in values if isinstance(value, (int, float))]
    if not filtered:
        return 0.0
    return sum(filtered) / float(len(filtered))


def _variance(values: Sequence[float]) -> float:
    filtered = [float(value) for value in values if isinstance(value, (int, float))]
    if len(filtered) < 2:
        return 0.0
    mu = _mean(filtered)
    return sum((value - mu) ** 2 for value in filtered) / float(len(filtered))


def _normalize_rate(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    if abs(value) > 1.0:
        if abs(value) <= 100.0:
            return round(value / 100.0, 6)
        return None
    return round(value, 6)


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, Mapping):
        value = value.get("value")
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _parse_iso_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
