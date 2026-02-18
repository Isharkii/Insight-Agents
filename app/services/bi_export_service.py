"""
app/services/bi_export_service.py

PowerBI-compatible export service.

Supports four datasets, each fully flattened for tabular consumption:

    records   — canonical_insight_records (primary insight data)
    kpis      — computed_kpis (one row per KPI metric, tall format)
    forecasts — forecast_metric (forecast_data JSONB flattened)
    risk      — business_risk_scores (risk_metadata JSONB flattened)

All nested JSONB structures are flattened one level deep using double-
underscore separators (e.g. ``forecast__month_1``, ``metadata__client_id``).
Second-level nesting and arrays are serialised to JSON strings so PowerBI
can optionally parse them with Power Query.

Date filters apply to the primary timestamp column of each dataset:
    records   — timestamp
    kpis      — period_end
    forecasts — period_end
    risk      — period_end

No transformation logic lives in the router.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterator

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models.canonical_insight_record import CanonicalInsightRecord
from db.models.computed_kpi import ComputedKPI
from forecast.repository import ForecastMetric
from risk.repository import BusinessRiskScore


_VALID_DATASETS: frozenset[str] = frozenset({"records", "kpis", "forecasts", "risk"})
_MAX_LIMIT: int = 100_000


# ---------------------------------------------------------------------------
# Export result container
# ---------------------------------------------------------------------------


@dataclass
class ExportResult:
    """
    Flat tabular data ready for CSV or JSON serialisation.

    Attributes
    ----------
    rows:   Flat dict per row; all values are JSON-safe scalars or strings.
    fields: Ordered column names; deterministic across calls for the same dataset.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# JSONB flattening helper
# ---------------------------------------------------------------------------


def _flatten_jsonb(value: Any, prefix: str, target: dict[str, Any]) -> None:
    """
    Flatten one JSON value into *target* under keys prefixed by *prefix*.

    Rules
    -----
    - ``None``   → nothing written (column omitted).
    - scalar     → ``{prefix: value}``
    - dict       → ``{prefix__{key}: child}`` for each child; second-level
                   dicts/lists are JSON-stringified.
    - list       → ``{prefix: json.dumps(value)}`` (serialised string)
    """
    if value is None:
        return
    if isinstance(value, dict):
        for k, v in value.items():
            col = f"{prefix}__{k}"
            if isinstance(v, (dict, list)):
                target[col] = json.dumps(v, default=str)
            else:
                target[col] = v
    elif isinstance(value, list):
        target[prefix] = json.dumps(value, default=str)
    else:
        target[prefix] = value


def _iso(dt: datetime | None) -> str | None:
    """Return UTC ISO-8601 string or None."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _collect_fields(rows: list[dict[str, Any]]) -> list[str]:
    """
    Union all keys across rows while preserving first-seen insertion order.
    Guarantees a deterministic, stable column list for CSV headers.
    """
    seen: dict[str, None] = {}
    for row in rows:
        for k in row:
            seen.setdefault(k, None)
    return list(seen)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class BIExportService:
    """
    Query and flatten analytics data for BI tool consumption.

    Every public method is read-only; no session commits are issued.
    The caller owns the session lifecycle.
    """

    def export(
        self,
        db: Session,
        *,
        dataset: str = "records",
        entity_name: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        limit: int = 10_000,
    ) -> ExportResult:
        """
        Run the export pipeline for *dataset* and return an :class:`ExportResult`.

        Parameters
        ----------
        db:          Active read-only SQLAlchemy session.
        dataset:     One of ``"records"``, ``"kpis"``, ``"forecasts"``, ``"risk"``.
        entity_name: Optional exact-match filter on ``entity_name``.
        date_from:   Inclusive lower bound on the date column (see module docstring).
        date_to:     Inclusive upper bound on the date column.
        limit:       Maximum rows returned; capped at ``_MAX_LIMIT``.

        Raises
        ------
        ValueError: When *dataset* is not one of the supported values.
        """
        if dataset not in _VALID_DATASETS:
            raise ValueError(
                f"Unknown dataset {dataset!r}. Valid: {sorted(_VALID_DATASETS)}"
            )
        safe_limit = max(1, min(limit, _MAX_LIMIT))
        handler = getattr(self, f"_export_{dataset}")
        return handler(
            db,
            entity_name=entity_name,
            date_from=date_from,
            date_to=date_to,
            limit=safe_limit,
        )

    # ------------------------------------------------------------------
    # Dataset handlers
    # ------------------------------------------------------------------

    def _export_records(
        self,
        db: Session,
        *,
        entity_name: str | None,
        date_from: date | None,
        date_to: date | None,
        limit: int,
    ) -> ExportResult:
        stmt = select(CanonicalInsightRecord).order_by(
            CanonicalInsightRecord.timestamp.desc()
        )
        if entity_name:
            stmt = stmt.where(CanonicalInsightRecord.entity_name == entity_name)
        if date_from:
            stmt = stmt.where(CanonicalInsightRecord.timestamp >= _date_start(date_from))
        if date_to:
            stmt = stmt.where(CanonicalInsightRecord.timestamp <= _date_end(date_to))
        stmt = stmt.limit(limit)

        rows = [self._flatten_record(r) for r in db.scalars(stmt)]
        return ExportResult(rows=rows, fields=_collect_fields(rows))

    def _export_kpis(
        self,
        db: Session,
        *,
        entity_name: str | None,
        date_from: date | None,
        date_to: date | None,
        limit: int,
    ) -> ExportResult:
        stmt = select(ComputedKPI).order_by(ComputedKPI.period_end.desc())
        if entity_name:
            stmt = stmt.where(ComputedKPI.entity_name == entity_name)
        if date_from:
            stmt = stmt.where(ComputedKPI.period_end >= _date_start(date_from))
        if date_to:
            stmt = stmt.where(ComputedKPI.period_end <= _date_end(date_to))
        stmt = stmt.limit(limit)

        # Expand each KPI record into one row per metric (tall format).
        rows: list[dict[str, Any]] = []
        for kpi in db.scalars(stmt):
            rows.extend(self._flatten_kpi(kpi))
        return ExportResult(rows=rows, fields=_collect_fields(rows))

    def _export_forecasts(
        self,
        db: Session,
        *,
        entity_name: str | None,
        date_from: date | None,
        date_to: date | None,
        limit: int,
    ) -> ExportResult:
        stmt = select(ForecastMetric).order_by(ForecastMetric.period_end.desc())
        if entity_name:
            stmt = stmt.where(ForecastMetric.entity_name == entity_name)
        if date_from:
            stmt = stmt.where(ForecastMetric.period_end >= _date_start(date_from))
        if date_to:
            stmt = stmt.where(ForecastMetric.period_end <= _date_end(date_to))
        stmt = stmt.limit(limit)

        rows = [self._flatten_forecast(f) for f in db.scalars(stmt)]
        return ExportResult(rows=rows, fields=_collect_fields(rows))

    def _export_risk(
        self,
        db: Session,
        *,
        entity_name: str | None,
        date_from: date | None,
        date_to: date | None,
        limit: int,
    ) -> ExportResult:
        stmt = select(BusinessRiskScore).order_by(BusinessRiskScore.period_end.desc())
        if entity_name:
            stmt = stmt.where(BusinessRiskScore.entity_name == entity_name)
        if date_from:
            stmt = stmt.where(BusinessRiskScore.period_end >= _date_start(date_from))
        if date_to:
            stmt = stmt.where(BusinessRiskScore.period_end <= _date_end(date_to))
        stmt = stmt.limit(limit)

        rows = [self._flatten_risk(r) for r in db.scalars(stmt)]
        return ExportResult(rows=rows, fields=_collect_fields(rows))

    # ------------------------------------------------------------------
    # Row flatteners
    # ------------------------------------------------------------------

    def _flatten_record(self, r: CanonicalInsightRecord) -> dict[str, Any]:
        """Flatten one canonical_insight_records row."""
        row: dict[str, Any] = {
            "id": str(r.id),
            "source_type": r.source_type,
            "entity_name": r.entity_name,
            "category": r.category,
            "metric_name": r.metric_name,
            "timestamp": _iso(r.timestamp),
            "region": r.region,
            "created_at": _iso(r.created_at),
        }
        _flatten_jsonb(r.metric_value, "metric_value", row)
        _flatten_jsonb(r.metadata_json, "metadata", row)
        return row

    def _flatten_kpi(self, k: ComputedKPI) -> list[dict[str, Any]]:
        """Expand one ComputedKPI row into one row per KPI metric (tall format)."""
        base: dict[str, Any] = {
            "kpi_id": str(k.id),
            "entity_name": k.entity_name,
            "period_start": _iso(k.period_start),
            "period_end": _iso(k.period_end),
            "created_at": _iso(k.created_at),
        }
        rows: list[dict[str, Any]] = []
        metrics: dict[str, Any] = k.computed_kpis or {}
        if not metrics:
            rows.append({**base, "metric_name": None, "metric_value": None, "metric_unit": None, "metric_error": None})
            return rows
        for metric_name, payload in metrics.items():
            row = {**base, "metric_name": metric_name}
            if isinstance(payload, dict):
                row["metric_value"] = payload.get("value")
                row["metric_unit"] = payload.get("unit")
                row["metric_error"] = payload.get("error")
            else:
                row["metric_value"] = payload
                row["metric_unit"] = None
                row["metric_error"] = None
            rows.append(row)
        return rows

    def _flatten_forecast(self, f: ForecastMetric) -> dict[str, Any]:
        """Flatten one forecast_metric row."""
        row: dict[str, Any] = {
            "id": str(f.id),
            "entity_name": f.entity_name,
            "metric_name": f.metric_name,
            "period_end": _iso(f.period_end),
            "created_at": _iso(f.created_at),
        }
        data: dict[str, Any] = f.forecast_data or {}
        # Scalar top-level fields
        for key in ("slope", "trend", "deviation_percentage"):
            row[f"forecast__{key}"] = data.get(key)
        # Nested forecast dict → flattened
        forecast_months = data.get("forecast")
        _flatten_jsonb(forecast_months, "forecast", row)
        return row

    def _flatten_risk(self, r: BusinessRiskScore) -> dict[str, Any]:
        """Flatten one business_risk_scores row."""
        row: dict[str, Any] = {
            "id": str(r.id),
            "entity_name": r.entity_name,
            "period_end": _iso(r.period_end),
            "risk_score": r.risk_score,
            "created_at": _iso(r.created_at),
        }
        _flatten_jsonb(r.risk_metadata, "risk_meta", row)
        return row


# ---------------------------------------------------------------------------
# Date window helpers
# ---------------------------------------------------------------------------


def _date_start(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)


def _date_end(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 23, 59, 59, 999999, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------


_service: BIExportService | None = None


def get_bi_export_service() -> BIExportService:
    global _service
    if _service is None:
        _service = BIExportService()
    return _service
