"""
agent/helpers/kpi_extraction.py

Pure data-extraction helpers for reading KPI payloads from agent state.
No I/O, no side-effects — only deterministic transformations.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import numpy as np

from agent.helpers.state_slimming import records_from_ref
from agent.graph_config import KPI_KEY_BY_BUSINESS_TYPE
from agent.nodes.node_result import payload_of, status_of
from agent.state import AgentState


def resolve_kpi_payload(state: AgentState) -> dict[str, Any]:
    """Select the appropriate KPI payload from state by business type."""
    business_type = str(state.get("business_type") or "").lower()
    preferred_key = KPI_KEY_BY_BUSINESS_TYPE.get(business_type)
    if preferred_key:
        preferred = state.get(preferred_key)
        if status_of(preferred) == "success":
            payload = payload_of(preferred)
            if isinstance(payload, dict):
                return payload

    for key in ("kpi_data", "saas_kpi_data", "ecommerce_kpi_data", "agency_kpi_data"):
        candidate = state.get(key)
        if status_of(candidate) == "success":
            payload = payload_of(candidate)
            if isinstance(payload, dict):
                return payload
    return {}


def extract_numeric_metric(
    computed_kpis: Mapping[str, Any],
    candidates: tuple[str, ...],
) -> float | None:
    """Pull the first valid numeric value from a set of candidate metric keys."""
    for key in candidates:
        raw = computed_kpis.get(key)
        value = raw.get("value") if isinstance(raw, dict) else raw
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            return numeric
    return None


def coerce_numeric(value: Any) -> float | None:
    """Coerce a raw value (or ``{"value": x}`` dict) to float, or ``None``."""
    if isinstance(value, Mapping):
        value = value.get("value")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(numeric):
        return float(numeric)
    return None


def parse_iso_datetime(value: Any) -> datetime | None:
    """Parse an ISO-8601 string to a UTC datetime, or ``None``."""
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


def resolve_period_bounds(kpi_payload: Mapping[str, Any]) -> tuple[datetime, datetime]:
    """Extract the most recent ``(period_start, period_end)`` from a KPI payload."""
    now = datetime.now(tz=timezone.utc)
    records = records_from_kpi_payload(kpi_payload)
    if isinstance(records, list) and records:
        bounds: list[tuple[datetime, datetime]] = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            start = parse_iso_datetime(record.get("period_start"))
            end = parse_iso_datetime(record.get("period_end"))
            if start is None or end is None:
                continue
            bounds.append((start, end))
        if bounds:
            return max(bounds, key=lambda item: item[1])

    start = parse_iso_datetime(kpi_payload.get("period_start"))
    end = parse_iso_datetime(kpi_payload.get("period_end"))
    if start is not None and end is not None:
        return start, end

    return now - timedelta(days=90), now


def dataset_confidence_from_state(state: AgentState) -> float:
    """Read ``dataset_confidence`` from state, clamped to [0, 1]."""
    raw = state.get("dataset_confidence")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 1.0
    return max(0.0, min(1.0, value))


def records_from_kpi_payload(kpi_payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Resolve raw KPI records from payload or compact ``record_ref`` cache."""
    if not isinstance(kpi_payload, Mapping):
        return []

    raw_records = kpi_payload.get("records")
    if isinstance(raw_records, list):
        return [record for record in raw_records if isinstance(record, Mapping)]

    record_ref = kpi_payload.get("record_ref")
    if isinstance(record_ref, str):
        return records_from_ref(record_ref)
    return []


def metric_series_from_kpi_payload(payload_or_records: Any) -> dict[str, list[float]]:
    """Build ``{metric_name: [values…]}`` from KPI payload or raw records."""
    if isinstance(payload_or_records, Mapping):
        precomputed = payload_or_records.get("metric_series")
        if isinstance(precomputed, Mapping):
            output: dict[str, list[float]] = {}
            for metric_name, values in precomputed.items():
                if not isinstance(values, list):
                    continue
                numeric_values: list[float] = []
                for value in values:
                    numeric = coerce_numeric(value)
                    if numeric is not None:
                        numeric_values.append(float(numeric))
                if numeric_values:
                    output[str(metric_name)] = numeric_values
            if output:
                return output
        records = records_from_kpi_payload(payload_or_records)
    elif isinstance(payload_or_records, list):
        records = payload_or_records
    else:
        records = []

    series_entries: dict[str, list[tuple[str, float]]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
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
            numeric = coerce_numeric(metric_entry)
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


def rows_from_kpi_payload(records: list[Any]) -> list[dict[str, Any]]:
    """Flatten KPI records into dimensional rows for role analytics."""
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        computed = record.get("computed_kpis")
        if not isinstance(computed, Mapping):
            continue
        for metric_name, metric_entry in computed.items():
            numeric = coerce_numeric(metric_entry)
            if numeric is None:
                continue
            rows.append(
                {
                    "role": record.get("role"),
                    "team": record.get("team"),
                    "channel": record.get("channel"),
                    "region": record.get("region"),
                    "product_line": record.get("product_line"),
                    "source_type": record.get("source_type"),
                    "metric_name": str(metric_name),
                    "metric_value": numeric,
                    "metadata_json": record.get("metadata_json"),
                }
            )
    return rows


def cohort_rows_from_kpi_payload(records: list[Any]) -> list[dict[str, Any]]:
    """Extract cohort-shaped rows from KPI payload records."""
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        computed = record.get("computed_kpis")
        if not isinstance(computed, Mapping):
            continue
        for metric_name, metric_entry in computed.items():
            numeric = coerce_numeric(metric_entry)
            if numeric is None:
                continue
            metadata = record.get("metadata_json")
            if not isinstance(metadata, Mapping):
                metadata = {}
            rows.append(
                {
                    "timestamp": record.get("period_end") or record.get("created_at"),
                    "metric_name": str(metric_name),
                    "metric_value": numeric,
                    "signup_month": record.get("signup_month") or metadata.get("signup_month"),
                    "acquisition_channel": (
                        record.get("acquisition_channel")
                        or metadata.get("acquisition_channel")
                        or record.get("channel")
                        or metadata.get("channel")
                    ),
                    "segment": record.get("segment") or metadata.get("segment"),
                    "metadata_json": dict(metadata),
                }
            )
    return rows
