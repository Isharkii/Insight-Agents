"""
agent/graph.py

LangGraph workflow assembly for the Insight Agent.

Pipeline status semantics
-------------------------
Each business type declares which state keys are *required* (the pipeline
cannot succeed without them) and which are *optional* (nice-to-have but
their absence only downgrades status to ``"partial"``).

    success → all required nodes produced a ``"success"`` envelope
    partial → all required nodes succeeded, but one or more optional nodes
              failed or were skipped
    failed  → at least one required node did not succeed
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import numpy as np
from langgraph.graph import END, START, StateGraph
from sqlalchemy import select

from agent.graph_config import (
    KPI_KEY_BY_BUSINESS_TYPE,
    graph_node_config_for_business_type,
)
from agent.nodes.agency_kpi_node import agency_kpi_fetch_node
from agent.nodes.business_router import business_router_node, route_by_business_type
from agent.nodes.ecommerce_kpi_node import ecommerce_kpi_fetch_node
from agent.nodes.forecast_node import forecast_fetch_node
from agent.nodes.intent import intent_node
from agent.nodes.kpi_node import kpi_fetch_node
from agent.nodes.llm_node import llm_node
from agent.nodes.node_result import failed, payload_of, skipped, status_of, success
from agent.nodes.prioritization_node import prioritization_node
from agent.nodes.risk_node import risk_node
from agent.nodes.saas_kpi_node import saas_kpi_fetch_node
from agent.nodes.synthesis_gate import synthesis_gate_node
from agent.state import AgentState
from app.services.category_registry import CategoryRegistryError, require_category_pack
from app.services.kpi_canonical_schema import (
    category_aliases_for_business_type,
    metric_aliases_for_business_type,
)
from app.services.cohort_analytics import DEFAULT_COHORT_KEYS, compute_cohort_analytics
from app.services.macro_context_service import build_macro_context
from app.services.role_dimension_analytics import build_role_dimension_summary
from app.services.role_performance_scoring import score_role_performance
from app.services.timeseries_factors import compute_timeseries_factors
from app.services.statistics.anomaly import detect_iqr_anomalies
from app.services.statistics.growth_engine import compute_growth_context
from app.services.statistics.multivariate import compute_multivariate_context
from app.services.statistics.normalization import (
    metric_statistics_config,
    rolling_mean,
    rolling_median,
    zscore_normalize,
)
from app.services.statistics.scenario_simulator import simulate_deterministic_scenarios
from db.models.canonical_insight_record import CanonicalInsightRecord
from db.session import SessionLocal

def derive_pipeline_status(state: AgentState) -> str:
    """Compute pipeline status from required/optional node classification.

    Rules
    -----
    * ``"success"`` — every *required* node produced ``status="success"``.
    * ``"partial"`` — all required nodes succeeded, but at least one *optional*
      node that is wired did not succeed.
    * ``"failed"``  — at least one required node is ``"skipped"`` or ``"failed"``.

    Unwired optional nodes (value is ``None`` in state because no graph
    node ever wrote to them) are silently ignored — they cannot drag the
    pipeline down to ``"partial"`` or ``"failed"``.
    """
    config = graph_node_config_for_business_type(str(state.get("business_type") or ""))
    required_keys = config.required
    optional_keys = config.optional

    # --- required nodes ---
    for key in required_keys:
        if status_of(state.get(key)) != "success":
            return "failed"

    # --- optional nodes (only those actually wired / populated) ---
    for key in optional_keys:
        value = state.get(key)
        # None means the node was never wired — skip entirely.
        if value is None:
            continue
        if status_of(value) != "success":
            return "partial"

    return "success"


def pipeline_status_node(state: AgentState) -> AgentState:
    """Pre-LLM node that computes ``pipeline_status`` from node outcomes."""
    return {**state, "pipeline_status": derive_pipeline_status(state)}


def _resolve_kpi_payload(state: AgentState) -> dict[str, Any]:
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


def _extract_numeric_metric(
    computed_kpis: Mapping[str, Any],
    candidates: tuple[str, ...],
) -> float | None:
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


def _parse_iso_datetime(value: Any) -> datetime | None:
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


def _resolve_period_bounds(kpi_payload: Mapping[str, Any]) -> tuple[datetime, datetime]:
    now = datetime.now(tz=timezone.utc)
    records = kpi_payload.get("records")
    if isinstance(records, list) and records:
        bounds: list[tuple[datetime, datetime]] = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            start = _parse_iso_datetime(record.get("period_start"))
            end = _parse_iso_datetime(record.get("period_end"))
            if start is None or end is None:
                continue
            bounds.append((start, end))
        if bounds:
            latest = max(bounds, key=lambda item: item[1])
            return latest

    start = _parse_iso_datetime(kpi_payload.get("period_start"))
    end = _parse_iso_datetime(kpi_payload.get("period_end"))
    if start is not None and end is not None:
        return start, end

    return now - timedelta(days=90), now


def _coerce_numeric(value: Any) -> float | None:
    if isinstance(value, Mapping):
        value = value.get("value")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(numeric):
        return float(numeric)
    return None


def _rows_from_kpi_payload(records: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        computed = record.get("computed_kpis")
        if not isinstance(computed, Mapping):
            continue
        for metric_name, metric_entry in computed.items():
            numeric = _coerce_numeric(metric_entry)
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


def _metric_series_from_kpi_payload(records: list[Any]) -> dict[str, list[float]]:
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


def _dataset_confidence_from_state(state: AgentState) -> float:
    raw = state.get("dataset_confidence")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 1.0
    return max(0.0, min(1.0, value))


def _cohort_rows_for_records(
    *,
    records: list[Any],
    entity_name: str,
    business_type: str,
    period_start: datetime,
    period_end: datetime,
) -> list[dict[str, Any]]:
    cohort_rows: list[dict[str, Any]] = []
    if entity_name:
        try:
            cohort_rows = _fetch_canonical_cohort_rows(
                entity_name=entity_name,
                business_type=business_type,
                period_start=period_start,
                period_end=period_end,
            )
        except Exception:
            cohort_rows = []
    if not cohort_rows:
        cohort_rows = _cohort_rows_from_kpi_payload(records)
    return cohort_rows


def _build_statistical_context(metric_series: Mapping[str, list[float]]) -> dict[str, Any]:
    if not metric_series:
        return {
            "status": "partial",
            "confidence_score": 0.5,
            "warnings": ["No metric series available for statistical context."],
            "metrics": {},
            "anomaly_summary": {
                "metric_count_with_anomalies": 0,
                "total_anomaly_points": 0,
                "metrics": [],
            },
        }

    metrics_payload: dict[str, Any] = {}
    warnings: list[str] = []
    partial_metrics = 0
    anomaly_metric_names: list[str] = []
    total_anomaly_points = 0

    for metric_name in sorted(metric_series):
        values = metric_series.get(metric_name, [])
        config = metric_statistics_config(metric_name)
        z_values = zscore_normalize(
            values,
            clip_abs=config.zscore_clip,
            zero_guard=config.zero_guard,
        )
        smoothed_mean = rolling_mean(values, window=config.smoothing_window)
        smoothed_median = rolling_median(values, window=config.smoothing_window)
        selected_smoothing = (
            smoothed_median if config.smoothing_method == "median" else smoothed_mean
        )
        anomaly = detect_iqr_anomalies(
            values,
            multiplier=config.anomaly_iqr_multiplier,
        )

        metric_status = "success"
        if len(values) < config.min_points:
            metric_status = "partial"
            partial_metrics += 1
            warnings.append(
                f"Metric '{metric_name}' has {len(values)} points; "
                f"minimum recommended is {config.min_points}."
            )

        anomaly_count = len(anomaly.get("anomaly_indexes", []))
        if anomaly_count > 0:
            anomaly_metric_names.append(metric_name)
            total_anomaly_points += anomaly_count

        metrics_payload[metric_name] = {
            "status": metric_status,
            "series_length": len(values),
            "applied_config": {
                "smoothing_window": config.smoothing_window,
                "smoothing_method": config.smoothing_method,
                "zscore_clip": config.zscore_clip,
                "anomaly_iqr_multiplier": config.anomaly_iqr_multiplier,
                "min_points": config.min_points,
            },
            "zscore": {
                "values": z_values,
                "clip_abs": config.zscore_clip,
            },
            "smoothing": {
                "mean": smoothed_mean,
                "median": smoothed_median,
                "selected_method": config.smoothing_method,
                "selected": selected_smoothing,
            },
            "anomaly": anomaly,
        }

    metric_count = max(1, len(metrics_payload))
    confidence_penalty = (partial_metrics / metric_count) * 0.4
    confidence_score = max(0.2, round(1.0 - confidence_penalty, 6))

    return {
        "status": "partial" if partial_metrics > 0 else "success",
        "confidence_score": confidence_score,
        "warnings": warnings,
        "metrics": metrics_payload,
        "anomaly_summary": {
            "metric_count_with_anomalies": len(anomaly_metric_names),
            "total_anomaly_points": total_anomaly_points,
            "metrics": sorted(anomaly_metric_names),
        },
    }


def _cohort_rows_from_kpi_payload(records: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        computed = record.get("computed_kpis")
        if not isinstance(computed, Mapping):
            continue
        for metric_name, metric_entry in computed.items():
            numeric = _coerce_numeric(metric_entry)
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


def _fetch_canonical_dimension_rows(
    *,
    entity_name: str,
    business_type: str,
    period_start: datetime,
    period_end: datetime,
) -> list[dict[str, Any]]:
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
            .order_by(CanonicalInsightRecord.timestamp.asc(), CanonicalInsightRecord.metric_name.asc())
        )
        query_rows = session.execute(stmt).all()

    result: list[dict[str, Any]] = []
    for row in query_rows:
        metric_value = _coerce_numeric(row.metric_value)
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


def _fetch_canonical_cohort_rows(
    *,
    entity_name: str,
    business_type: str,
    period_start: datetime,
    period_end: datetime,
) -> list[dict[str, Any]]:
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
            .order_by(CanonicalInsightRecord.timestamp.asc(), CanonicalInsightRecord.metric_name.asc())
        )
        query_rows = session.execute(stmt).all()

    result: list[dict[str, Any]] = []
    for row in query_rows:
        metric_value = _coerce_numeric(row.metric_value)
        if metric_value is None:
            continue
        metadata = row.metadata_json if isinstance(row.metadata_json, Mapping) else {}
        result.append(
            {
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                "metric_name": row.metric_name,
                "metric_value": metric_value,
                "signup_month": metadata.get("signup_month"),
                "acquisition_channel": metadata.get("acquisition_channel") or metadata.get("channel"),
                "segment": metadata.get("segment") or metadata.get("customer_segment"),
                "metadata_json": dict(metadata),
            }
        )
    return result


def _fetch_macro_context_rows(
    *,
    entity_name: str,
    business_type: str,
    period_start: datetime,
    period_end: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
            .order_by(CanonicalInsightRecord.timestamp.asc(), CanonicalInsightRecord.metric_name.asc())
        )
        inflation_query_rows = session.execute(inflation_stmt).all()

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
                CanonicalInsightRecord.entity_name == entity_name,
                CanonicalInsightRecord.category.in_(categories),
                CanonicalInsightRecord.timestamp >= lookback_start,
                CanonicalInsightRecord.timestamp <= period_end,
            )
            .order_by(CanonicalInsightRecord.timestamp.asc(), CanonicalInsightRecord.metric_name.asc())
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
    return inflation_rows, benchmark_rows


def growth_engine_node(state: AgentState) -> AgentState:
    """Compute deterministic growth signals from KPI time series."""
    try:
        kpi_payload = _resolve_kpi_payload(state)
        records = kpi_payload.get("records") if isinstance(kpi_payload, dict) else None
        if not isinstance(records, list) or not records:
            return {**state, "growth_data": skipped("kpi_unavailable", {"records": 0})}

        metric_series = _metric_series_from_kpi_payload(records)
        if not metric_series:
            return {**state, "growth_data": skipped("series_unavailable", {"records": len(records)})}

        business_type = str(state.get("business_type") or "").strip().lower()
        aliases = metric_aliases_for_business_type(business_type)
        revenue_candidates = aliases.get("recurring_revenue", ("recurring_revenue",))
        growth_context = compute_growth_context(
            metric_series,
            preferred_metric_candidates=revenue_candidates,
        )

        dataset_confidence = _dataset_confidence_from_state(state)
        warnings = [str(item) for item in growth_context.get("warnings", [])]
        if dataset_confidence < 1.0:
            warnings.append(
                f"Dataset confidence reduced growth reliability ({dataset_confidence:.2f})."
            )
        confidence = min(
            dataset_confidence,
            float(growth_context.get("confidence_score") or 1.0),
        )
        growth_data = success(growth_context, warnings=warnings, confidence_score=confidence)
        return {**state, "growth_data": growth_data}
    except Exception as exc:  # noqa: BLE001
        return {**state, "growth_data": failed(str(exc), {"stage": "growth_engine"})}


def timeseries_factors_node(state: AgentState) -> AgentState:
    """Compute deterministic time-series factor flags."""
    try:
        kpi_payload = _resolve_kpi_payload(state)
        records = kpi_payload.get("records") if isinstance(kpi_payload, dict) else None
        if not isinstance(records, list) or not records:
            return {
                **state,
                "timeseries_factors_data": skipped("kpi_unavailable", {"records": 0}),
            }

        metric_series = _metric_series_from_kpi_payload(records)
        if not metric_series:
            return {
                **state,
                "timeseries_factors_data": skipped("series_unavailable", {"records": len(records)}),
            }

        growth_payload = payload_of(state.get("growth_data")) or {}
        primary_metric = str(growth_payload.get("primary_metric") or "").strip()
        if primary_metric not in metric_series:
            primary_metric = sorted(metric_series, key=lambda name: (-len(metric_series[name]), name))[0]

        factors = compute_timeseries_factors(metric_series.get(primary_metric, []))
        warnings: list[str] = []
        if str(factors.get("volatility_regime") or "") == "insufficient_history":
            warnings.append("Insufficient history for volatility regime detection.")
        if str(factors.get("cycle_state") or "") == "insufficient_history":
            warnings.append("Insufficient history for cycle state detection.")

        dataset_confidence = _dataset_confidence_from_state(state)
        base_confidence = 0.7 if warnings else 1.0
        confidence = min(dataset_confidence, base_confidence)
        payload = {
            "primary_metric": primary_metric,
            "series_points": len(metric_series.get(primary_metric, [])),
            "factors": factors,
        }
        return {
            **state,
            "timeseries_factors_data": success(
                payload,
                warnings=warnings,
                confidence_score=confidence,
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            **state,
            "timeseries_factors_data": failed(str(exc), {"stage": "timeseries_factors"}),
        }


def cohort_analytics_node(state: AgentState) -> AgentState:
    """Compute cohort analytics when cohort keys are available."""
    try:
        kpi_payload = _resolve_kpi_payload(state)
        records = kpi_payload.get("records") if isinstance(kpi_payload, dict) else None
        if not isinstance(records, list) or not records:
            return {**state, "cohort_data": skipped("kpi_unavailable", {"records": 0})}

        business_type = str(state.get("business_type") or "").strip().lower()
        entity_name = str(
            kpi_payload.get("fetched_for")
            or state.get("entity_name")
            or ""
        ).strip()
        period_start, period_end = _resolve_period_bounds(kpi_payload)
        cohort_rows = _cohort_rows_for_records(
            records=records,
            entity_name=entity_name,
            business_type=business_type,
            period_start=period_start,
            period_end=period_end,
        )
        if not cohort_rows:
            return {**state, "cohort_data": skipped("cohort_not_applicable", {"records": len(records)})}

        aliases = metric_aliases_for_business_type(business_type)
        active_candidates = aliases.get("active_customer_count", ("active_customer_count",))
        churn_candidates = aliases.get("churned_customer_count", ("churned_customer_count",))
        cohort = compute_cohort_analytics(
            cohort_rows,
            cohort_keys=DEFAULT_COHORT_KEYS,
            active_metric_names=active_candidates,
            churn_metric_names=churn_candidates,
        )

        dataset_confidence = _dataset_confidence_from_state(state)
        warnings = [str(item) for item in cohort.get("warnings", [])]
        confidence = min(dataset_confidence, float(cohort.get("confidence_score") or 1.0))
        return {
            **state,
            "cohort_data": success(
                cohort,
                warnings=warnings,
                confidence_score=confidence,
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {**state, "cohort_data": failed(str(exc), {"stage": "cohort_analytics"})}


def _dependency_value(
    *,
    source: str,
    agg: Mapping[str, Any],
    extra: Mapping[str, Any],
    default: Any,
) -> Any:
    prefix, _, key = source.partition(".")
    prefix = prefix.strip().lower()
    key = key.strip()
    if prefix == "agg":
        value = agg.get(key)
        return default if value is None else value
    if prefix == "extra":
        return extra.get(key, default)
    return default


def _is_missing(value: Any, *, missing_when: str) -> bool:
    if missing_when == "is_empty":
        if value is None:
            return True
        if isinstance(value, (str, bytes)):
            return not value
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0
        return False
    return value is None


def category_formula_node(state: AgentState) -> AgentState:
    """Run category-pack deterministic formula using registry bindings."""
    try:
        business_type = str(state.get("business_type") or "").strip().lower()
        try:
            pack = require_category_pack(business_type)
        except CategoryRegistryError:
            pack = require_category_pack("general_timeseries")

        kpi_payload = _resolve_kpi_payload(state)
        records = kpi_payload.get("records") if isinstance(kpi_payload, dict) else None
        if not isinstance(records, list) or not records:
            return {
                **state,
                "category_formula_data": skipped("kpi_unavailable", {"category": pack.name}),
            }

        metric_series = _metric_series_from_kpi_payload(records)
        aliases = pack.metric_aliases
        revenue_aliases = aliases.get("recurring_revenue", ("recurring_revenue",))
        active_aliases = aliases.get("active_customer_count", ("active_customer_count",))
        churn_aliases = aliases.get("churned_customer_count", ("churned_customer_count",))

        def _pick_series(candidates: tuple[str, ...]) -> list[float]:
            for candidate in candidates:
                if candidate in metric_series:
                    return list(metric_series[candidate])
            return []

        revenue_series = _pick_series(revenue_aliases)
        active_series = _pick_series(active_aliases)
        churn_series = _pick_series(churn_aliases)
        agg_values: dict[str, Any] = {
            "subscription_revenues": list(revenue_series),
            "active_customers": int(round(active_series[-1])) if active_series else 0,
            "lost_customers": int(round(churn_series[-1])) if churn_series else 0,
            "previous_revenue": float(revenue_series[-2]) if len(revenue_series) >= 2 else 0.0,
        }
        extra_values: dict[str, Any] = {}

        formula_inputs: dict[str, Any] = {}
        for key, binding in pack.formula_input_bindings.items():
            formula_inputs[key] = _dependency_value(
                source=binding.source,
                agg=agg_values,
                extra=extra_values,
                default=binding.default,
            )
        metrics = pack.formula.calculate(formula_inputs)

        optional_missing: list[str] = []
        required_missing: list[str] = []
        for metric_name, dependencies in pack.validity_rules.items():
            missing = []
            for dependency in dependencies:
                value = _dependency_value(
                    source=dependency.source,
                    agg=agg_values,
                    extra=extra_values,
                    default=None,
                )
                if _is_missing(value, missing_when=dependency.missing_when):
                    missing.append(dependency.source)
            if missing:
                if metric_name in pack.optional_signals:
                    optional_missing.append(metric_name)
                else:
                    required_missing.append(metric_name)

        metric_payload: dict[str, Any] = {}
        for name, value in metrics.items():
            metric_payload[name] = {
                "value": _coerce_numeric(value),
                "status": (
                    "missing_optional"
                    if name in optional_missing
                    else ("missing_required" if name in required_missing else "success")
                ),
            }

        warnings: list[str] = []
        if optional_missing:
            warnings.append(f"Optional metrics unavailable: {sorted(optional_missing)}")
        if required_missing:
            warnings.append(f"Required formula dependencies missing: {sorted(required_missing)}")

        dataset_confidence = _dataset_confidence_from_state(state)
        optional_ratio = (len(optional_missing) / max(1, len(pack.optional_signals))) if pack.optional_signals else 0.0
        confidence = max(0.2, min(dataset_confidence, round(1.0 - (optional_ratio * 0.4), 6)))
        payload = {
            "category": pack.name,
            "required_fields": list(pack.required_inputs),
            "canonical_metric_aliases": {k: list(v) for k, v in pack.metric_aliases.items()},
            "optional_missing": sorted(optional_missing),
            "metrics": metric_payload,
        }
        return {
            **state,
            "category_formula_data": success(
                payload,
                warnings=warnings,
                confidence_score=confidence,
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            **state,
            "category_formula_data": failed(str(exc), {"stage": "category_formula"}),
        }


def multivariate_scenario_node(state: AgentState) -> AgentState:
    """Compute statistical, multivariate, and deterministic scenario payloads."""
    try:
        kpi_payload = _resolve_kpi_payload(state)
        records = kpi_payload.get("records") if isinstance(kpi_payload, dict) else None
        if not isinstance(records, list) or not records:
            return {
                **state,
                "multivariate_scenario_data": skipped("kpi_unavailable", {"records": 0}),
            }

        metric_series = _metric_series_from_kpi_payload(records)
        if not metric_series:
            return {
                **state,
                "multivariate_scenario_data": skipped("series_unavailable", {"records": len(records)}),
            }

        business_type = str(state.get("business_type") or "").strip().lower()
        aliases = metric_aliases_for_business_type(business_type)
        revenue_candidates = aliases.get("recurring_revenue", ("recurring_revenue",))
        growth_context = payload_of(state.get("growth_data")) or compute_growth_context(
            metric_series,
            preferred_metric_candidates=revenue_candidates,
        )
        statistical_context = _build_statistical_context(metric_series)

        entity_name = str(
            kpi_payload.get("fetched_for")
            or state.get("entity_name")
            or ""
        ).strip()
        period_start, period_end = _resolve_period_bounds(kpi_payload)
        cohort_rows = _cohort_rows_for_records(
            records=records,
            entity_name=entity_name,
            business_type=business_type,
            period_start=period_start,
            period_end=period_end,
        )
        multivariate_context = compute_multivariate_context(
            metric_series,
            segment_rows=cohort_rows,
            preferred_metric_candidates=revenue_candidates,
        )
        scenario_simulation = simulate_deterministic_scenarios(
            metric_series,
            growth_context=growth_context,
            statistical_context=statistical_context,
            multivariate_context=multivariate_context,
            preferred_metric_candidates=revenue_candidates,
        )
        payload = {
            "statistical_context": statistical_context,
            "multivariate_context": multivariate_context,
            "scenario_simulation": scenario_simulation,
        }
        confidence_candidates = [
            float(statistical_context.get("confidence_score") or 0.5),
            float(multivariate_context.get("confidence_score") or 0.5),
            float(scenario_simulation.get("base_confidence") or 0.5),
            _dataset_confidence_from_state(state),
        ]
        confidence = max(0.2, min(confidence_candidates))
        warnings: list[str] = []
        warnings.extend(str(item) for item in statistical_context.get("warnings", []))
        warnings.extend(str(item) for item in multivariate_context.get("warnings", []))
        warnings.extend(str(item) for item in scenario_simulation.get("warnings", []))
        return {
            **state,
            "multivariate_scenario_data": success(
                payload,
                warnings=warnings,
                confidence_score=confidence,
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            **state,
            "multivariate_scenario_data": failed(
                str(exc),
                {"stage": "multivariate_scenario"},
            ),
        }


def role_analytics_node(state: AgentState) -> AgentState:
    """
    Role analytics stage inserted between KPI and risk.

    Internal execution:
      KPI -> Role Aggregation -> Statistical Engine -> Role Scoring.
    """
    try:
        kpi_payload = _resolve_kpi_payload(state)
        records = kpi_payload.get("records") if isinstance(kpi_payload, dict) else None
        if not isinstance(records, list) or not records:
            role_analytics_data = skipped("kpi_unavailable", {"records": 0})
            return {**state, "segmentation": role_analytics_data}

        business_type = str(state.get("business_type") or "").strip().lower()
        entity_name = str(
            kpi_payload.get("fetched_for")
            or state.get("entity_name")
            or ""
        ).strip()

        period_start, period_end = _resolve_period_bounds(kpi_payload)
        canonical_rows: list[dict[str, Any]] = []
        if entity_name:
            canonical_rows = _fetch_canonical_dimension_rows(
                entity_name=entity_name,
                business_type=business_type,
                period_start=period_start,
                period_end=period_end,
            )

        if not canonical_rows:
            canonical_rows = _rows_from_kpi_payload(records)
        if not canonical_rows:
            role_analytics_data = skipped("dimension_values_missing", {"records": len(records)})
            return {**state, "segmentation": role_analytics_data}

        summary = build_role_dimension_summary(canonical_rows, top_n=3)
        if not summary.get("records_used"):
            role_analytics_data = skipped("dimension_values_missing", {"records": len(canonical_rows)})
            return {**state, "segmentation": role_analytics_data}

        inflation_rows: list[dict[str, Any]] = []
        benchmark_rows: list[dict[str, Any]] = []
        if entity_name:
            try:
                inflation_rows, benchmark_rows = _fetch_macro_context_rows(
                    entity_name=entity_name,
                    business_type=business_type,
                    period_start=period_start,
                    period_end=period_end,
                )
            except Exception:
                inflation_rows = []
                benchmark_rows = []

        aliases = metric_aliases_for_business_type(business_type)
        revenue_candidates = aliases.get("recurring_revenue", ("recurring_revenue",))
        macro_context = build_macro_context(
            kpi_payload=kpi_payload,
            inflation_rows=inflation_rows,
            benchmark_rows=benchmark_rows,
            metric_candidates=revenue_candidates,
        )

        cohort_rows = _cohort_rows_for_records(
            records=records,
            entity_name=entity_name,
            business_type=business_type,
            period_start=period_start,
            period_end=period_end,
        )
        active_candidates = aliases.get("active_customer_count", ("active_customer_count",))
        churn_candidates = aliases.get("churned_customer_count", ("churned_customer_count",))
        metric_series = _metric_series_from_kpi_payload(records)
        growth_context = payload_of(state.get("growth_data")) or compute_growth_context(
            metric_series,
            preferred_metric_candidates=revenue_candidates,
        )
        timeseries_factors = payload_of(state.get("timeseries_factors_data")) or {}
        cohort_analytics = payload_of(state.get("cohort_data")) or compute_cohort_analytics(
            cohort_rows,
            cohort_keys=DEFAULT_COHORT_KEYS,
            active_metric_names=active_candidates,
            churn_metric_names=churn_candidates,
        )
        category_formula = payload_of(state.get("category_formula_data")) or {}
        multivariate_bundle = payload_of(state.get("multivariate_scenario_data")) or {}
        statistical_context = multivariate_bundle.get("statistical_context")
        if not isinstance(statistical_context, Mapping):
            statistical_context = _build_statistical_context(metric_series)
        multivariate_context = multivariate_bundle.get("multivariate_context")
        if not isinstance(multivariate_context, Mapping):
            multivariate_context = compute_multivariate_context(
                metric_series,
                segment_rows=cohort_rows,
                preferred_metric_candidates=revenue_candidates,
            )
        scenario_simulation = multivariate_bundle.get("scenario_simulation")
        if not isinstance(scenario_simulation, Mapping):
            scenario_simulation = simulate_deterministic_scenarios(
                metric_series,
                growth_context=growth_context,
                statistical_context=statistical_context,
                multivariate_context=multivariate_context,
                preferred_metric_candidates=revenue_candidates,
            )
        statistical_context = {
            **statistical_context,
            "derived": {
                "multivariate": multivariate_context,
                "scenario_simulation": scenario_simulation,
            },
        }

        scoring_input: dict[str, dict[str, Any]] = {}
        for dimension, payload in summary.get("by_dimension", {}).items():
            contributors = payload.get("contributors", [])
            if not isinstance(contributors, list):
                continue
            for contributor in contributors:
                if not isinstance(contributor, Mapping):
                    continue
                name = str(contributor.get("name") or "").strip()
                if not name:
                    continue
                share = float(contributor.get("contribution_share") or 0.0)
                key = f"{dimension}:{name}"
                scoring_input[key] = {
                    "growth_rate": [share],
                    "efficiency_metric": [share],
                    "contribution_weight": [share],
                    "stability_series": [share],
                }

        role_scores = score_role_performance(scoring_input, category=business_type)
        node_warnings: list[str] = []
        for key in (
            "growth_data",
            "timeseries_factors_data",
            "cohort_data",
            "category_formula_data",
            "multivariate_scenario_data",
        ):
            envelope = state.get(key)
            if isinstance(envelope, Mapping):
                node_warnings.extend(str(item) for item in envelope.get("warnings", []) if str(item).strip())

        confidence_inputs = [_dataset_confidence_from_state(state)]
        for key in (
            "growth_data",
            "timeseries_factors_data",
            "cohort_data",
            "category_formula_data",
            "multivariate_scenario_data",
        ):
            envelope = state.get(key)
            if isinstance(envelope, Mapping):
                raw_conf = envelope.get("confidence_score")
                try:
                    conf = float(raw_conf)
                except (TypeError, ValueError):
                    continue
                confidence_inputs.append(max(0.0, min(1.0, conf)))

        role_analytics_data = success(
            {
                "dimensions": summary.get("dimensions", []),
                "top_contributors": summary.get("top_contributors", []),
                "laggards": summary.get("laggards", []),
                "dependency_concentration": summary.get("dependency_concentration", {}),
                "by_dimension": summary.get("by_dimension", {}),
                "records_scanned": summary.get("records_scanned", 0),
                "records_used": summary.get("records_used", 0),
                "role_scoring": role_scores,
                "macro_context": macro_context,
                "cohort_analytics": cohort_analytics,
                "timeseries_factors": timeseries_factors,
                "category_formula": category_formula,
                "statistical_context": statistical_context,
                "growth_context": growth_context,
                "multivariate_context": multivariate_context,
                "scenario_simulation": scenario_simulation,
            },
            warnings=node_warnings,
            confidence_score=min(confidence_inputs),
        )
        return {**state, "segmentation": role_analytics_data}

    except Exception as exc:  # noqa: BLE001
        role_analytics_data = failed(str(exc), {"stage": "role_analytics_node"})
        return {**state, "segmentation": role_analytics_data}


def build_graph():
    """Build and compile the Insight Agent LangGraph workflow."""
    graph = StateGraph(AgentState)

    graph.add_node("intent", intent_node)
    graph.add_node("business_router", business_router_node)
    graph.add_node("kpi_fetch", kpi_fetch_node)
    graph.add_node("saas_kpi_fetch", saas_kpi_fetch_node)
    graph.add_node("ecommerce_kpi_fetch", ecommerce_kpi_fetch_node)
    graph.add_node("agency_kpi_fetch", agency_kpi_fetch_node)
    graph.add_node("growth_engine", growth_engine_node)
    graph.add_node("timeseries_factors", timeseries_factors_node)
    graph.add_node("cohort_analytics", cohort_analytics_node)
    graph.add_node("category_formulas", category_formula_node)
    graph.add_node("multivariate_scenario", multivariate_scenario_node)
    graph.add_node("role_analytics", role_analytics_node)
    graph.add_node("forecast_fetch", forecast_fetch_node)
    graph.add_node("risk", risk_node)
    graph.add_node("prioritization", prioritization_node)
    graph.add_node("pipeline_status", pipeline_status_node)
    graph.add_node("synthesis_gate", synthesis_gate_node)
    graph.add_node("llm", llm_node)

    graph.add_edge(START, "intent")
    graph.add_edge("intent", "business_router")
    graph.add_conditional_edges(
        "business_router",
        route_by_business_type,
        {
            "kpi_fetch": "kpi_fetch",
            "saas_kpi_fetch": "saas_kpi_fetch",
            "ecommerce_kpi_fetch": "ecommerce_kpi_fetch",
            "agency_kpi_fetch": "agency_kpi_fetch",
        },
    )
    graph.add_edge("kpi_fetch", "growth_engine")
    graph.add_edge("saas_kpi_fetch", "growth_engine")
    graph.add_edge("ecommerce_kpi_fetch", "growth_engine")
    graph.add_edge("agency_kpi_fetch", "growth_engine")
    graph.add_edge("growth_engine", "timeseries_factors")
    graph.add_edge("timeseries_factors", "cohort_analytics")
    graph.add_edge("cohort_analytics", "category_formulas")
    graph.add_edge("category_formulas", "multivariate_scenario")
    graph.add_edge("multivariate_scenario", "role_analytics")
    graph.add_edge("role_analytics", "forecast_fetch")
    graph.add_edge("forecast_fetch", "risk")
    graph.add_edge("risk", "prioritization")
    graph.add_edge("prioritization", "pipeline_status")
    graph.add_edge("pipeline_status", "synthesis_gate")

    # Conditional edge: if required signals failed, skip LLM entirely
    def _route_after_gate(state: AgentState) -> str:
        if state.get("synthesis_blocked"):
            return "end"
        return "llm"

    graph.add_conditional_edges(
        "synthesis_gate",
        _route_after_gate,
        {"llm": "llm", "end": END},
    )
    graph.add_edge("llm", END)

    return graph.compile()


insight_graph = build_graph()
