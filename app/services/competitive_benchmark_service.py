"""
app/services/competitive_benchmark_service.py

Deterministic peer benchmarking snapshot builder used by API routers.

This service wires:
  - peer sourcing (same category window, exclude client, optional size band)
  - validated directional ranking (MetricComparisonSpec + rank_competitive)
  - confidence-weighted composite scoring (score_composite)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.services.kpi_canonical_schema import category_aliases_for_business_type
from app.services.ranking_engine import MetricComparisonSpec, rank_competitive
from app.services.scoring_engine import score_composite
from db.models.canonical_insight_record import CanonicalInsightRecord
from db.repositories.kpi_repository import KPIRepository
from risk.repository import RiskRepository

_DEFAULT_WINDOW_DAYS = 180
_DEFAULT_MAX_PEERS = 25
_MIN_PEERS_FOR_SIZE_BAND = 3
_SIZE_BAND_LOWER = 0.5
_SIZE_BAND_UPPER = 2.0

_DEFAULT_AGGREGATION = "monthly_avg"
_DEFAULT_WINDOW_ALIGNMENT = "trailing_6m"

_MOMENTUM_GROWTH_CANDIDATES = (
    "growth_rate",
    "projected_growth",
    "revenue_growth_rate",
)
_MRR_CANDIDATES = (
    "mrr",
    "revenue",
    "total_revenue",
    "retainer_revenue",
)


@dataclass(frozen=True)
class _ObservationBundle:
    observations: dict[str, dict[str, Any]]
    scalar_values: dict[str, float]
    confidences: dict[str, float]


def build_competitive_benchmark_snapshot(
    *,
    db: Session,
    entity_name: str,
    business_type: str,
    window_days: int = _DEFAULT_WINDOW_DAYS,
    max_peers: int = _DEFAULT_MAX_PEERS,
) -> dict[str, Any]:
    """Build a deterministic competitive benchmark snapshot for one entity."""
    resolved_entity = str(entity_name or "").strip()
    if not resolved_entity:
        return {"status": "skipped", "reason": "missing_entity_name"}

    now = datetime.now(tz=timezone.utc)
    period_start = now - timedelta(days=max(30, int(window_days)))
    period_end = now

    category_aliases = tuple(category_aliases_for_business_type(business_type))
    peer_entities = _source_peer_entities(
        db=db,
        entity_name=resolved_entity,
        category_aliases=category_aliases,
        period_start=period_start,
        period_end=period_end,
    )

    repo = KPIRepository(db)
    all_rows = repo.get_kpis_by_period(
        period_start=period_start,
        period_end=period_end,
        entity_name=None,
    )
    latest_rows = _latest_rows_by_entity(all_rows)
    client_latest = latest_rows.get(resolved_entity)
    if client_latest is None:
        return {
            "status": "partial",
            "reason": "client_kpi_unavailable",
            "entity_name": resolved_entity,
            "business_type": business_type,
            "window": {
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "aggregation": _DEFAULT_AGGREGATION,
                "window_alignment": _DEFAULT_WINDOW_ALIGNMENT,
            },
            "peer_selection": {
                "sourcing_rule": "same_category_same_window_excluding_client",
                "categories": list(category_aliases),
                "peer_candidates": peer_entities,
                "selected_peers": [],
            },
        }

    metric_specs = _build_metric_specs(client_latest)
    client_bundle = _build_observations(client_latest, metric_specs)
    if not client_bundle.observations:
        return {
            "status": "partial",
            "reason": "no_valid_client_metrics",
            "entity_name": resolved_entity,
            "business_type": business_type,
            "window": {
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "aggregation": _DEFAULT_AGGREGATION,
                "window_alignment": _DEFAULT_WINDOW_ALIGNMENT,
            },
            "metric_comparison_specs": _metric_specs_payload(metric_specs),
            "peer_selection": {
                "sourcing_rule": "same_category_same_window_excluding_client",
                "categories": list(category_aliases),
                "peer_candidates": peer_entities,
                "selected_peers": [],
            },
        }

    selected_peer_entities = _select_peer_entities(
        entity_name=resolved_entity,
        peer_candidates=peer_entities,
        latest_rows=latest_rows,
        client_scalars=client_bundle.scalar_values,
        max_peers=max_peers,
    )
    if not selected_peer_entities:
        return {
            "status": "partial",
            "reason": "insufficient_peers_in_aligned_window",
            "entity_name": resolved_entity,
            "business_type": business_type,
            "window": {
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "aggregation": _DEFAULT_AGGREGATION,
                "window_alignment": _DEFAULT_WINDOW_ALIGNMENT,
            },
            "metric_comparison_specs": _metric_specs_payload(metric_specs),
            "peer_selection": {
                "sourcing_rule": "same_category_same_window_excluding_client",
                "categories": list(category_aliases),
                "peer_candidates": peer_entities,
                "selected_peers": [],
                "excluded_entity": resolved_entity,
                "size_band": {
                    "enabled": True,
                    "lower_multiplier": _SIZE_BAND_LOWER,
                    "upper_multiplier": _SIZE_BAND_UPPER,
                },
            },
        }

    competitor_observations: dict[str, dict[str, dict[str, Any]]] = {}
    benchmark_data: dict[str, list[float]] = {name: [] for name in client_bundle.scalar_values}
    benchmark_confidences: dict[str, list[float]] = {name: [] for name in client_bundle.scalar_values}

    risk_repo = RiskRepository()
    client_risk = _safe_risk_score(risk_repo.get_latest_risk(db, resolved_entity))
    if client_risk is not None:
        client_bundle.scalar_values["risk_score"] = client_risk
        client_bundle.confidences["risk_score"] = 1.0
        benchmark_data.setdefault("risk_score", [])
        benchmark_confidences.setdefault("risk_score", [])

    for peer_name in selected_peer_entities:
        peer_row = latest_rows.get(peer_name)
        if peer_row is None:
            continue
        peer_bundle = _build_observations(peer_row, metric_specs)
        if peer_bundle.observations:
            competitor_observations[peer_name] = peer_bundle.observations

        for metric_name, metric_value in peer_bundle.scalar_values.items():
            if metric_name not in benchmark_data:
                continue
            benchmark_data[metric_name].append(metric_value)
            benchmark_confidences[metric_name].append(
                peer_bundle.confidences.get(metric_name, 1.0),
            )

        if "risk_score" in benchmark_data:
            peer_risk = _safe_risk_score(risk_repo.get_latest_risk(db, peer_name))
            if peer_risk is not None:
                benchmark_data["risk_score"].append(peer_risk)
                benchmark_confidences["risk_score"].append(1.0)

    # Keep only metrics with at least one benchmark observation.
    benchmark_data = {k: v for k, v in benchmark_data.items() if v}
    benchmark_confidences = {k: v for k, v in benchmark_confidences.items() if v}
    client_scalars = {
        k: v
        for k, v in client_bundle.scalar_values.items()
        if k in benchmark_data
    }
    client_confidences = {
        k: v
        for k, v in client_bundle.confidences.items()
        if k in benchmark_data
    }
    if not benchmark_data or not client_scalars:
        return {
            "status": "partial",
            "reason": "no_comparable_peer_metrics_after_validation",
            "entity_name": resolved_entity,
            "business_type": business_type,
            "window": {
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "aggregation": _DEFAULT_AGGREGATION,
                "window_alignment": _DEFAULT_WINDOW_ALIGNMENT,
            },
            "metric_comparison_specs": _metric_specs_payload(metric_specs),
            "peer_selection": {
                "sourcing_rule": "same_category_same_window_excluding_client",
                "categories": list(category_aliases),
                "peer_candidates": peer_entities,
                "selected_peers": selected_peer_entities,
                "excluded_entity": resolved_entity,
                "size_band": {
                    "enabled": True,
                    "lower_multiplier": _SIZE_BAND_LOWER,
                    "upper_multiplier": _SIZE_BAND_UPPER,
                },
            },
        }

    ranking = rank_competitive(
        client_name=resolved_entity,
        client_metrics=client_bundle.observations,
        competitor_metrics=competitor_observations,
        metric_specs=metric_specs,
    ).model_dump()

    composite = score_composite(
        client_metrics=client_scalars,
        benchmark_data=benchmark_data,
        metric_directions={
            name: spec.direction
            for name, spec in metric_specs.items()
            if name in client_scalars
        },
        use_confidence_weighting=True,
        client_confidences=client_confidences,
        benchmark_confidences=benchmark_confidences,
        metric_series=_metric_series_for_entity(
            all_rows,
            resolved_entity,
        ),
        growth_metric_candidates=_MOMENTUM_GROWTH_CANDIDATES,
        mrr_metric_candidates=_MRR_CANDIDATES,
        risk_metric_candidates=("risk_score",),
        stability_metric_candidates=tuple(client_scalars.keys()) or ("mrr",),
    ).model_dump()

    return {
        "status": "success",
        "entity_name": resolved_entity,
        "business_type": business_type,
        "window": {
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "aggregation": _DEFAULT_AGGREGATION,
            "window_alignment": _DEFAULT_WINDOW_ALIGNMENT,
        },
        "metric_comparison_specs": _metric_specs_payload(metric_specs),
        "peer_selection": {
            "sourcing_rule": "same_category_same_window_excluding_client",
            "categories": list(category_aliases),
            "peer_candidates": peer_entities,
            "selected_peers": selected_peer_entities,
            "excluded_entity": resolved_entity,
            "size_band": {
                "enabled": True,
                "lower_multiplier": _SIZE_BAND_LOWER,
                "upper_multiplier": _SIZE_BAND_UPPER,
            },
        },
        "ranking": ranking,
        "composite": composite,
    }


def _source_peer_entities(
    *,
    db: Session,
    entity_name: str,
    category_aliases: tuple[str, ...],
    period_start: datetime,
    period_end: datetime,
) -> list[str]:
    stmt = (
        select(CanonicalInsightRecord.entity_name)
        .where(
            CanonicalInsightRecord.category.in_(category_aliases),
            CanonicalInsightRecord.timestamp >= period_start,
            CanonicalInsightRecord.timestamp <= period_end,
        )
        .distinct()
    )
    values = db.scalars(stmt).all()
    peers = sorted(
        {
            str(value).strip()
            for value in values
            if str(value).strip() and str(value).strip() != entity_name
        },
    )
    return peers


def _latest_rows_by_entity(rows: list[Any]) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for row in rows:
        name = str(getattr(row, "entity_name", "") or "").strip()
        if not name:
            continue
        current = latest.get(name)
        if current is None or _row_period_end(row) > _row_period_end(current):
            latest[name] = row
    return latest


def _select_peer_entities(
    *,
    entity_name: str,
    peer_candidates: list[str],
    latest_rows: dict[str, Any],
    client_scalars: dict[str, float],
    max_peers: int,
) -> list[str]:
    peers = [name for name in peer_candidates if name in latest_rows and name != entity_name]
    if not peers:
        peers = sorted(
            [
                name
                for name in latest_rows
                if name != entity_name
            ],
        )

    client_size = _size_metric_value(client_scalars)
    if client_size is None or client_size <= 0.0:
        return peers[: max(1, max_peers)]

    lower = client_size * _SIZE_BAND_LOWER
    upper = client_size * _SIZE_BAND_UPPER
    size_filtered: list[str] = []
    for peer_name in peers:
        peer_size = _size_metric_from_row(latest_rows[peer_name])
        if peer_size is None:
            continue
        if lower <= peer_size <= upper:
            size_filtered.append(peer_name)

    selected = size_filtered if len(size_filtered) >= _MIN_PEERS_FOR_SIZE_BAND else peers
    return selected[: max(1, max_peers)]


def _size_metric_value(values: dict[str, float]) -> float | None:
    for metric_name in _MRR_CANDIDATES:
        value = values.get(metric_name)
        if value is not None and math.isfinite(value):
            return float(value)
    return None


def _size_metric_from_row(row: Any) -> float | None:
    computed = getattr(row, "computed_kpis", {}) or {}
    for metric_name in _MRR_CANDIDATES:
        if metric_name not in computed:
            continue
        numeric = _entry_value(computed.get(metric_name))
        if numeric is None:
            continue
        return numeric
    return None


def _build_metric_specs(row: Any) -> dict[str, MetricComparisonSpec]:
    computed = getattr(row, "computed_kpis", {}) or {}
    specs: dict[str, MetricComparisonSpec] = {}
    for metric_name, entry in computed.items():
        value = _entry_value(entry)
        if value is None:
            continue
        metric = str(metric_name).strip()
        if not metric:
            continue
        unit = _entry_unit(metric, entry)
        specs[metric] = MetricComparisonSpec(
            metric=metric,
            direction=_direction_for_metric(metric),
            unit=unit,
            scale=_scale_for_entry(entry),
            aggregation=_DEFAULT_AGGREGATION,
            window_alignment=_DEFAULT_WINDOW_ALIGNMENT,
        )
    return specs


def _build_observations(
    row: Any,
    metric_specs: dict[str, MetricComparisonSpec],
) -> _ObservationBundle:
    computed = getattr(row, "computed_kpis", {}) or {}
    observations: dict[str, dict[str, Any]] = {}
    scalars: dict[str, float] = {}
    confidences: dict[str, float] = {}

    for metric_name, spec in metric_specs.items():
        if metric_name not in computed:
            continue
        entry = computed.get(metric_name)
        numeric = _entry_value(entry)
        if numeric is None:
            continue

        observations[metric_name] = {
            "value": numeric,
            "unit": _entry_unit(metric_name, entry),
            "scale": _scale_for_entry(entry),
            "aggregation": _DEFAULT_AGGREGATION,
            "window_alignment": _DEFAULT_WINDOW_ALIGNMENT,
        }
        scalars[metric_name] = numeric
        confidences[metric_name] = _entry_confidence(entry)

    return _ObservationBundle(
        observations=observations,
        scalar_values=scalars,
        confidences=confidences,
    )


def _metric_series_for_entity(rows: list[Any], entity_name: str) -> dict[str, list[float]]:
    series_entries: dict[str, list[tuple[datetime, float]]] = {}
    for row in rows:
        if str(getattr(row, "entity_name", "") or "").strip() != entity_name:
            continue
        period_end = _row_period_end(row)
        computed = getattr(row, "computed_kpis", {}) or {}
        for metric_name, entry in computed.items():
            numeric = _entry_value(entry)
            if numeric is None:
                continue
            key = str(metric_name).strip()
            if not key:
                continue
            series_entries.setdefault(key, []).append((period_end, numeric))

    output: dict[str, list[float]] = {}
    for metric_name, values in series_entries.items():
        ordered = sorted(values, key=lambda item: item[0])
        output[metric_name] = [float(value) for _, value in ordered]
    return output


def _metric_specs_payload(specs: dict[str, MetricComparisonSpec]) -> dict[str, dict[str, Any]]:
    return {
        name: spec.model_dump()
        for name, spec in specs.items()
    }


def _row_period_end(row: Any) -> datetime:
    value = getattr(row, "period_end", None)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _entry_value(entry: Any) -> float | None:
    raw = entry.get("value") if isinstance(entry, dict) else entry
    try:
        numeric = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return float(numeric)


def _entry_unit(metric_name: str, entry: Any) -> str:
    if isinstance(entry, dict):
        unit = str(entry.get("unit") or "").strip()
        if unit:
            return unit
    if "rate" in metric_name or "churn" in metric_name:
        return "rate"
    return "currency"


def _scale_for_entry(entry: Any) -> str:
    if not isinstance(entry, dict):
        return "raw"
    unit = str(entry.get("unit") or "").strip().lower()
    if unit in {"percent", "percentage", "%"}:
        return "percentage"
    if unit == "rate":
        return "ratio"
    return "raw"


def _direction_for_metric(metric_name: str) -> str:
    lowered = metric_name.strip().lower()
    if "churn" in lowered or "cac" in lowered or "risk" in lowered:
        return "lower_is_better"
    return "higher_is_better"


def _entry_confidence(entry: Any) -> float:
    if not isinstance(entry, dict):
        return 1.0
    is_valid = entry.get("is_valid")
    if isinstance(is_valid, bool):
        return 1.0 if is_valid else 0.35
    error = str(entry.get("error") or "").strip()
    if error:
        return 0.35
    return 1.0


def _safe_risk_score(record: Any) -> float | None:
    if record is None:
        return None
    raw = getattr(record, "risk_score", None)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return max(0.0, min(100.0, value))
