"""
agent/helpers/state_slimming.py

State-size optimization utilities for KPI payloads.

Design goals:
1) Keep graph state compact by storing derived metrics only.
2) Avoid duplicating raw record lists across node transitions/checkpoints.
3) Retain backward compatibility via optional on-demand raw record lookup.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Mapping
from uuid import uuid4

_CACHE_LOCK = Lock()
_RAW_KPI_CACHE: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
_RAW_KPI_CACHE_MAX = 128


def slim_kpi_payload(
    records: list[dict[str, Any]],
    *,
    fetched_for: str | None,
    period_start: str,
    period_end: str,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Build a compact KPI payload for graph state.

    The payload intentionally excludes raw records to reduce state size.
    Raw records are stored once in an in-process cache and referenced by key.
    """
    ordered = _ordered_records(records)
    record_ref = _store_raw_records(ordered)
    metric_series = _metric_series_from_records(ordered)
    latest_kpis = _latest_computed_kpis(ordered)

    effective_period_start = period_start
    effective_period_end = period_end
    if ordered:
        first = ordered[0]
        last = ordered[-1]
        first_start = str(first.get("period_start") or "").strip()
        last_end = str(last.get("period_end") or "").strip()
        if first_start:
            effective_period_start = first_start
        if last_end:
            effective_period_end = last_end

    metric_names = [
        str(item).strip()
        for item in (metrics or sorted(metric_series.keys()))
        if str(item).strip()
    ]

    return {
        "state_mode": "derived_only",
        "record_ref": record_ref,
        "record_count": len(ordered),
        "fetched_for": fetched_for,
        "period_start": effective_period_start,
        "period_end": effective_period_end,
        "metrics": metric_names,
        "metric_series": metric_series,
        "latest_computed_kpis": latest_kpis,
        "cache_generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def records_from_ref(record_ref: str | None) -> list[dict[str, Any]]:
    """Get raw KPI records from cache for a compact payload reference."""
    key = str(record_ref or "").strip()
    if not key:
        return []
    with _CACHE_LOCK:
        records = _RAW_KPI_CACHE.get(key)
        if records is None:
            return []
        # Refresh LRU position.
        _RAW_KPI_CACHE.move_to_end(key)
        return list(records)


def clear_raw_kpi_cache() -> None:
    """Testing utility to clear cached raw KPI records."""
    with _CACHE_LOCK:
        _RAW_KPI_CACHE.clear()


def _store_raw_records(records: list[dict[str, Any]]) -> str:
    key = str(uuid4())
    with _CACHE_LOCK:
        _RAW_KPI_CACHE[key] = list(records)
        _RAW_KPI_CACHE.move_to_end(key)
        while len(_RAW_KPI_CACHE) > _RAW_KPI_CACHE_MAX:
            _RAW_KPI_CACHE.popitem(last=False)
    return key


def _ordered_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _sort_key(record: Mapping[str, Any]) -> tuple[str, str]:
        period_end = str(record.get("period_end") or "").strip()
        created_at = str(record.get("created_at") or "").strip()
        return period_end, created_at

    usable = [record for record in records if isinstance(record, Mapping)]
    return sorted(usable, key=_sort_key)


def _metric_series_from_records(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    series_entries: dict[str, list[tuple[str, float]]] = {}
    for record in records:
        computed = record.get("computed_kpis")
        if not isinstance(computed, Mapping):
            continue
        sort_key = str(
            record.get("period_end")
            or record.get("created_at")
            or record.get("period_start")
            or ""
        ).strip()
        for metric_name, metric_entry in computed.items():
            numeric = _coerce_numeric(metric_entry)
            if numeric is None:
                continue
            metric = str(metric_name).strip()
            if not metric:
                continue
            series_entries.setdefault(metric, []).append((sort_key, float(numeric)))

    output: dict[str, list[float]] = {}
    for metric_name, entries in series_entries.items():
        ordered = sorted(entries, key=lambda item: item[0])
        output[metric_name] = [float(value) for _, value in ordered]
    return output


def _latest_computed_kpis(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    latest = records[-1]
    computed = latest.get("computed_kpis")
    return dict(computed) if isinstance(computed, Mapping) else {}


def _coerce_numeric(value: Any) -> float | None:
    if isinstance(value, Mapping):
        value = value.get("value")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
