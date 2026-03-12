"""
agent/helpers/canonical_queries.py

Database query helpers that fetch canonical insight records for graph nodes.
Extracted from graph.py to keep SQL access in a single module.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Mapping

from sqlalchemy import select

from agent.helpers.kpi_extraction import (
    coerce_numeric,
    cohort_rows_from_kpi_payload,
    resolve_period_bounds,
)
from app.services.kpi_canonical_schema import (
    category_aliases_for_business_type,
    metric_aliases_for_business_type,
)
from db.models.canonical_insight_record import CanonicalInsightRecord
from db.session import SessionLocal


def fetch_canonical_dimension_rows(
    *,
    entity_name: str,
    business_type: str,
    period_start: datetime,
    period_end: datetime,
) -> list[dict[str, Any]]:
    """Query canonical records for role/dimension analytics."""
    categories = category_aliases_for_business_type(business_type)
    aliases = metric_aliases_for_business_type(business_type)
    revenue_metrics = aliases.get("recurring_revenue", ("recurring_revenue",))

    with SessionLocal() as session:
        stmt = (
            select(
                CanonicalInsightRecord.role,
                CanonicalInsightRecord.region,
                CanonicalInsightRecord.source_type,
                CanonicalInsightRecord.metric_name,
                CanonicalInsightRecord.metric_value,
                CanonicalInsightRecord.metadata_json,
            )
            .where(
                CanonicalInsightRecord.entity_name == entity_name,
                CanonicalInsightRecord.category.in_(categories),
                CanonicalInsightRecord.metric_name.in_(revenue_metrics),
                CanonicalInsightRecord.timestamp >= period_start,
                CanonicalInsightRecord.timestamp <= period_end,
            )
            .order_by(
                CanonicalInsightRecord.timestamp.asc(),
                CanonicalInsightRecord.metric_name.asc(),
            )
        )
        query_rows = session.execute(stmt).all()

    result: list[dict[str, Any]] = []
    for row in query_rows:
        metric_value = coerce_numeric(row.metric_value)
        if metric_value is None:
            continue
        metadata = row.metadata_json if isinstance(row.metadata_json, Mapping) else {}
        result.append(
            {
                "role": row.role,
                "region": row.region,
                "source_type": row.source_type,
                "metric_name": row.metric_name,
                "metric_value": metric_value,
                "team": metadata.get("team"),
                "channel": metadata.get("channel"),
                "product_line": metadata.get("product_line"),
                "metadata_json": dict(metadata),
            }
        )
    return result


def fetch_canonical_cohort_rows(
    *,
    entity_name: str,
    business_type: str,
    period_start: datetime,
    period_end: datetime,
) -> list[dict[str, Any]]:
    """Query canonical records for cohort analytics with 1-year lookback."""
    categories = category_aliases_for_business_type(business_type)
    lookback_start = period_start - timedelta(days=365)
    with SessionLocal() as session:
        stmt = (
            select(
                CanonicalInsightRecord.metric_name,
                CanonicalInsightRecord.metric_value,
                CanonicalInsightRecord.timestamp,
                CanonicalInsightRecord.metadata_json,
            )
            .where(
                CanonicalInsightRecord.entity_name == entity_name,
                CanonicalInsightRecord.category.in_(categories),
                CanonicalInsightRecord.timestamp >= lookback_start,
                CanonicalInsightRecord.timestamp <= period_end,
            )
            .order_by(
                CanonicalInsightRecord.timestamp.asc(),
                CanonicalInsightRecord.metric_name.asc(),
            )
        )
        query_rows = session.execute(stmt).all()

    result: list[dict[str, Any]] = []
    for row in query_rows:
        metric_value = coerce_numeric(row.metric_value)
        if metric_value is None:
            continue
        metadata = row.metadata_json if isinstance(row.metadata_json, Mapping) else {}
        result.append(
            {
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                "metric_name": row.metric_name,
                "metric_value": metric_value,
                "signup_month": metadata.get("signup_month"),
                "acquisition_channel": (
                    metadata.get("acquisition_channel") or metadata.get("channel")
                ),
                "segment": metadata.get("segment") or metadata.get("customer_segment"),
                "metadata_json": dict(metadata),
            }
        )
    return result


def cohort_rows_for_records(
    *,
    records: list[Any],
    entity_name: str,
    business_type: str,
    period_start: datetime,
    period_end: datetime,
) -> list[dict[str, Any]]:
    """Get cohort rows from DB (preferred) or fall back to KPI payload."""
    cohort_rows: list[dict[str, Any]] = []
    if entity_name:
        try:
            cohort_rows = fetch_canonical_cohort_rows(
                entity_name=entity_name,
                business_type=business_type,
                period_start=period_start,
                period_end=period_end,
            )
        except Exception:
            cohort_rows = []
    if not cohort_rows:
        cohort_rows = cohort_rows_from_kpi_payload(records)
    return cohort_rows


def fetch_macro_context_rows(
    *,
    entity_name: str,
    business_type: str,
    period_start: datetime,
    period_end: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch inflation + benchmark peer rows for macro context."""
    lookback_start = period_start - timedelta(days=730)
    categories = category_aliases_for_business_type(business_type)

    with SessionLocal() as session:
        inflation_stmt = (
            select(
                CanonicalInsightRecord.entity_name,
                CanonicalInsightRecord.category,
                CanonicalInsightRecord.source_type,
                CanonicalInsightRecord.metric_name,
                CanonicalInsightRecord.metric_value,
                CanonicalInsightRecord.timestamp,
                CanonicalInsightRecord.metadata_json,
            )
            .where(
                CanonicalInsightRecord.category == "macro",
                CanonicalInsightRecord.timestamp >= lookback_start,
                CanonicalInsightRecord.timestamp <= period_end,
            )
            .order_by(
                CanonicalInsightRecord.timestamp.asc(),
                CanonicalInsightRecord.metric_name.asc(),
            )
        )
        inflation_query_rows = session.execute(inflation_stmt).all()

        client_band_stmt = (
            select(CanonicalInsightRecord.metadata_json)
            .where(
                CanonicalInsightRecord.entity_name == entity_name,
                CanonicalInsightRecord.category.in_(categories),
                CanonicalInsightRecord.timestamp >= period_start,
                CanonicalInsightRecord.timestamp <= period_end,
            )
            .order_by(CanonicalInsightRecord.timestamp.desc())
            .limit(500)
        )
        client_band_rows = session.execute(client_band_stmt).all()
        client_size_band = infer_size_band_from_metadata_rows(client_band_rows)

        benchmark_stmt = (
            select(
                CanonicalInsightRecord.entity_name,
                CanonicalInsightRecord.category,
                CanonicalInsightRecord.source_type,
                CanonicalInsightRecord.metric_name,
                CanonicalInsightRecord.metric_value,
                CanonicalInsightRecord.timestamp,
                CanonicalInsightRecord.metadata_json,
            )
            .where(
                CanonicalInsightRecord.entity_name != entity_name,
                CanonicalInsightRecord.category.in_(categories),
                CanonicalInsightRecord.timestamp >= period_start,
                CanonicalInsightRecord.timestamp <= period_end,
            )
            .order_by(
                CanonicalInsightRecord.timestamp.asc(),
                CanonicalInsightRecord.metric_name.asc(),
            )
        )
        benchmark_query_rows = session.execute(benchmark_stmt).all()

    def _serialize(rows: list[Any]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for row in rows:
            metadata = row.metadata_json if isinstance(row.metadata_json, Mapping) else {}
            serialized.append(
                {
                    "entity_name": row.entity_name,
                    "category": row.category,
                    "source_type": row.source_type,
                    "metric_name": row.metric_name,
                    "metric_value": row.metric_value,
                    "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                    "metadata_json": dict(metadata),
                }
            )
        return serialized

    inflation_rows = _serialize(inflation_query_rows)
    benchmark_rows = _serialize(benchmark_query_rows)
    benchmark_rows = filter_peers_by_size_band(
        benchmark_rows=benchmark_rows,
        client_size_band=client_size_band,
    )
    return inflation_rows, benchmark_rows


def infer_size_band_from_metadata_rows(rows: list[Any]) -> str | None:
    """Resolve a client size band from metadata rows."""
    candidate_keys = (
        "size_band",
        "company_size_band",
        "employee_size_band",
        "org_size_band",
        "firm_size_band",
    )
    for row in rows:
        metadata = None
        try:
            metadata = row[0]
        except Exception:
            metadata = getattr(row, "metadata_json", None)
        if not isinstance(metadata, Mapping):
            continue
        for key in candidate_keys:
            value = metadata.get(key)
            if value is None:
                continue
            normalized = str(value).strip().lower()
            if normalized:
                return normalized
    return None


def filter_peers_by_size_band(
    *,
    benchmark_rows: list[dict[str, Any]],
    client_size_band: str | None,
) -> list[dict[str, Any]]:
    """Best-effort size-band alignment for peer benchmarks."""
    if not client_size_band:
        return benchmark_rows

    candidate_keys = (
        "size_band",
        "company_size_band",
        "employee_size_band",
        "org_size_band",
        "firm_size_band",
    )
    matched: list[dict[str, Any]] = []
    for row in benchmark_rows:
        metadata = row.get("metadata_json")
        if not isinstance(metadata, Mapping):
            continue
        for key in candidate_keys:
            value = metadata.get(key)
            if value is None:
                continue
            normalized = str(value).strip().lower()
            if normalized == client_size_band:
                matched.append(row)
                break
    if matched:
        return matched
    return benchmark_rows
