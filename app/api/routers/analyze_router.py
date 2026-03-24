"""
app/api/routers/analyze_router.py

Insight generation endpoint — exposes the full LangGraph pipeline as an API.
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agent.graph import insight_graph
from agent.nodes.intent import intent_node
from agent.signal_integrity import UnifiedSignalIntegrity
from app.failure_codes import (
    INGESTION_VALIDATION,
    INTERNAL_FAILURE,
    SCHEMA_CONFLICT,
    build_error_detail,
)
from app.security.dependencies import (
    assert_entity_allowed_for_tenant,
    require_security_context,
)
from app.security.models import SecurityContext
from app.repositories.dataset_repository import DatasetRepository
from app.services.category_registry import (
    get_processing_strategy,
    primary_metric_for_business_type,
    supported_categories,
)
from app.domain.canonical_insight import IngestionStatus
from app.services.csv_ingestion_service import (
    CSVIngestionService,
    get_csv_ingestion_service,
)
from app.services.kpi_canonical_schema import (
    category_aliases_for_business_type,
    infer_analytics_strategy_from_categories,
)
from app.services.dataset_hash import compute_dataset_hash
from app.services.kpi_orchestrator import ANALYTICS_VERSION, KPIOrchestrator, KPIRunResult
from db.models.canonical_insight_record import CanonicalInsightRecord
from db.repositories.kpi_repository import KPIRepository
from db.session import get_db
from forecast.orchestrator import ForecastOrchestrator
from llm_synthesis.schema import InsightOutput

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analysis"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _should_skip_computation(
    db: Session,
    entity_name: str,
    current_hash: str,
) -> bool:
    """Skip KPI computation only when ALL stored rows match the current
    analytics version AND dataset hash.

    Returns ``False`` (= must recompute) when:
    - no rows exist at all (first run)
    - any row has a different ``analytics_version`` (code changed)
    - any row has a different ``dataset_hash`` (data changed)
    - any row has NULL version/hash (pre-versioning legacy row)
    """
    repo = KPIRepository(db)
    rows = repo.get_kpis_by_period(
        period_start=datetime(1970, 1, 1, tzinfo=timezone.utc),
        period_end=datetime.now(tz=timezone.utc),
        entity_name=entity_name,
    )
    if not rows:
        return False

    return all(
        r.analytics_version == ANALYTICS_VERSION
        and r.dataset_hash == current_hash
        for r in rows
    )


def _has_any_computed_kpis(db: Session, entity_name: str) -> bool:
    """Simple existence check — used as a safety net before graph invocation."""
    repo = KPIRepository(db)
    rows = repo.get_kpis_by_period(
        period_start=datetime(1970, 1, 1, tzinfo=timezone.utc),
        period_end=datetime.now(tz=timezone.utc),
        entity_name=entity_name,
    )
    return bool(rows)


def _generate_monthly_windows(
    period_start: datetime,
    period_end: datetime,
) -> list[tuple[datetime, datetime]]:
    """Split a date range into calendar-month windows.

    Each window is half-open: [first-of-month 00:00 UTC, first-of-next-month 00:00 UTC).
    The aggregation layer uses ``timestamp >= start AND timestamp < end`` so
    boundary rows are never double-counted across adjacent windows.

    The loop uses ``<=`` so that when ``period_end`` falls exactly on a month
    boundary (e.g. June 1st), a dedicated window is still created for that month.
    """
    from dateutil.relativedelta import relativedelta

    start = period_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)

    windows: list[tuple[datetime, datetime]] = []
    while start <= period_end:
        end = start + relativedelta(months=1)
        windows.append((start, end))
        start = end
    return windows


def _ensure_analytics_data(
    *,
    entity_name: str,
    processing_strategy: str,
    db: Session,
) -> None:
    """Run KPI computation + forecast for *entity_name* if not already present.

    Computes KPIs per calendar month so that downstream graph nodes,
    dashboards, and derived signals receive a proper time-series of
    data points instead of a single collapsed aggregate.
    """
    data_start, data_end = _resolve_kpi_period_from_entity_data(
        db=db,
        entity_name=entity_name,
        processing_strategy=processing_strategy,
    )

    # Compute dataset hash once for the full period
    cat_aliases = category_aliases_for_business_type(processing_strategy)
    current_hash = compute_dataset_hash(
        db, entity_name, cat_aliases, data_start, data_end,
    )

    if _should_skip_computation(db, entity_name, current_hash):
        logger.info(
            "Analyze: skipping KPI computation entity=%r — version=%d hash=%s matches",
            entity_name, ANALYTICS_VERSION, current_hash[:12],
        )
        return

    # --- Monthly KPI computation ---
    monthly_windows = _generate_monthly_windows(data_start, data_end)
    orchestrator = KPIOrchestrator()
    all_results: list[KPIRunResult] = []

    for win_start, win_end in monthly_windows:
        try:
            kpi_result = orchestrator.run(
                entity_name=entity_name,
                business_type=processing_strategy,
                period_start=win_start,
                period_end=win_end,
                db=db,
                dataset_hash=current_hash,
            )
            all_results.append(kpi_result)
        except Exception:
            logger.warning(
                "Analyze: KPI computation failed entity=%r window=[%s, %s]",
                entity_name,
                win_start.isoformat(),
                win_end.isoformat(),
                exc_info=True,
            )

    logger.info(
        "Analyze: KPI computed entity=%r processing_strategy=%r windows=%d succeeded=%d",
        entity_name,
        processing_strategy,
        len(monthly_windows),
        len(all_results),
    )

    # --- Forecast generation (uses all monthly primary-metric values) ---
    try:
        metric_name = primary_metric_for_business_type(processing_strategy)
        values: list[float] = []
        for result in all_results:
            entry = result.metrics.get(metric_name, {})
            value = entry.get("value")
            if isinstance(value, (int, float)):
                values.append(float(value))

        forecast_result = ForecastOrchestrator(db).generate_forecast(
            entity_name=entity_name,
            metric_name=metric_name,
            values=values,
        )
        if "error" in forecast_result:
            logger.info(
                "Analyze: forecast deferred entity=%r metric=%r: %s",
                entity_name,
                metric_name,
                forecast_result["error"],
            )
        else:
            db.commit()
            logger.info(
                "Analyze: forecast generated entity=%r metric=%r points=%d",
                entity_name,
                metric_name,
                len(values),
            )
    except Exception:
        db.rollback()
        logger.warning(
            "Analyze: forecast generation failed entity=%r",
            entity_name,
            exc_info=True,
        )


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _derive_kpi_period_bounds(
    *,
    earliest: datetime | None,
    latest: datetime | None,
    now: datetime,
) -> tuple[datetime, datetime]:
    """
    Build KPI computation bounds from dataset timestamps.

    Uses a rolling 90-day window anchored to the latest available record.
    Falls back to [now-90d, now] when no timestamps are available.
    If data starts within 90 days of latest, keep the true earliest bound.
    """
    if latest is None:
        return now - timedelta(days=90), now

    latest_utc = _ensure_utc(latest)
    window_start = latest_utc - timedelta(days=90)
    if earliest is None:
        return window_start, latest_utc

    earliest_utc = _ensure_utc(earliest)
    if earliest_utc > latest_utc:
        return window_start, latest_utc
    return max(earliest_utc, window_start), latest_utc


def _resolve_kpi_period_from_entity_data(
    *,
    db: Session,
    entity_name: str,
    processing_strategy: str,
) -> tuple[datetime, datetime]:
    """
    Resolve KPI period from canonical records for the entity + strategy categories.
    """
    categories = category_aliases_for_business_type(processing_strategy)
    row = db.execute(
        select(
            func.min(CanonicalInsightRecord.timestamp),
            func.max(CanonicalInsightRecord.timestamp),
        ).where(
            CanonicalInsightRecord.entity_name == entity_name,
            CanonicalInsightRecord.category.in_(categories),
        )
    ).one()
    earliest, latest = row[0], row[1]
    now = datetime.now(tz=timezone.utc)
    period_start, period_end = _derive_kpi_period_bounds(
        earliest=earliest,
        latest=latest,
        now=now,
    )
    return period_start, period_end


def _extract_primary_metric_values(
    kpi_result: KPIRunResult | None,
    metric_name: str,
) -> list[float]:
    """Return metric values from a KPI result for the forecast pipeline."""
    if kpi_result is None:
        return []
    entry = kpi_result.metrics.get(metric_name, {})
    value = entry.get("value")
    if not isinstance(value, (int, float)):
        return []
    return [float(value)]


def _has_kpi_data(state: dict[str, Any]) -> bool:
    """Check whether the pipeline produced any KPI records.

    Supports both legacy payloads (with a ``records`` list) and slim
    payloads produced by ``slim_kpi_payload`` (with ``record_count``).
    """
    for key, payload in state.items():
        if key != "kpi_data" and not key.endswith("_kpi_data"):
            continue
        if isinstance(payload, dict):
            data = payload.get("payload") if "status" in payload else payload
            if isinstance(data, dict):
                # Slim payload: record_count > 0 means records exist.
                if data.get("record_count", 0) > 0:
                    return True
                # Legacy payload: explicit records list.
                records = data.get("records")
                if isinstance(records, list) and records:
                    return True
    return False


def _extract_distinct_entity_names_from_canonical_records(
    *,
    db: Session,
    created_since: datetime,
) -> list[str]:
    """Return distinct entity_name values for rows inserted after created_since."""
    stmt = (
        select(CanonicalInsightRecord.entity_name)
        .where(CanonicalInsightRecord.created_at >= created_since)
        .distinct()
    )
    values = db.scalars(stmt).all()
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _count_records_per_entity(
    *,
    db: Session,
    entities: list[str],
    created_since: datetime | None = None,
) -> dict[str, int]:
    """Return record counts per entity for deterministic target selection.

    When *created_since* is ``None`` all records for the given entities are
    counted — this is the correct behaviour when rows were deduped and no
    new records were created during this ingestion cycle.
    """

    stmt = (
        select(
            CanonicalInsightRecord.entity_name,
            func.count().label("cnt"),
        )
        .where(CanonicalInsightRecord.entity_name.in_(entities))
        .group_by(CanonicalInsightRecord.entity_name)
    )
    if created_since is not None:
        stmt = stmt.where(CanonicalInsightRecord.created_at >= created_since)
    return {str(row[0]).strip(): int(row[1]) for row in db.execute(stmt).all()}


def _infer_analytics_strategy_from_dataset(*, db: Session, entity_name: str) -> str | None:
    """Infer KPI strategy from categories present for the resolved entity."""
    repository = DatasetRepository(db)
    categories = [
        str(value).strip()
        for value in repository.get_distinct_categories(entity_name=entity_name)
        if str(value).strip()
    ]
    return infer_analytics_strategy_from_categories(categories)


def _estimate_dataset_confidence(
    *,
    db: Session,
    entity_name: str,
    processing_strategy: str | None = None,
) -> float:
    """
    Estimate dataset confidence from stored ingestion provenance in metadata_json.
    Falls back to 1.0 when no provenance confidence is found.
    """
    categories = category_aliases_for_business_type(processing_strategy or "")
    stmt = select(CanonicalInsightRecord.metadata_json).where(
        CanonicalInsightRecord.entity_name == entity_name,
    )
    if categories:
        stmt = stmt.where(CanonicalInsightRecord.category.in_(categories))
    rows = db.execute(stmt.limit(1000)).all()

    confidences: list[float] = []
    for row in rows:
        metadata = row[0]
        if not isinstance(metadata, dict):
            continue
        ingestion = metadata.get("_ingestion")
        if not isinstance(ingestion, dict):
            continue
        raw = ingestion.get("schema_confidence")
        try:
            confidence = float(raw)
        except (TypeError, ValueError):
            continue
        confidences.append(max(0.0, min(1.0, confidence)))
    if not confidences:
        return 1.0
    return max(0.0, min(1.0, sum(confidences) / len(confidences)))


def _is_benign_no_valid_records(ingest_summary: Any) -> bool:
    """
    Return True when ingestion produced only a synthetic no_valid_records marker.

    This commonly happens when all uploaded rows already exist and are skipped
    by dedupe-on-conflict.
    """
    if ingest_summary.rows_failed != 0:
        return False
    errors = ingest_summary.validation_errors or []
    if len(errors) != 1:
        return False
    code = str(errors[0].code or "").strip().lower()
    return code == "no_valid_records"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def _extract_pipeline_signals(state: dict[str, Any]) -> dict[str, Any]:
    """Extract intermediate pipeline signals from graph state for the frontend.

    Returns a flat dict of deterministic signals that the frontend can
    render as charts, tables, and detail panels.
    """
    from agent.nodes.node_result import confidence_of, payload_of, status_of

    signals: dict[str, Any] = {}

    # ── Risk ──
    risk_envelope = state.get("risk_data")
    risk_payload = payload_of(risk_envelope) or {}
    signals["risk"] = {
        "status": status_of(risk_envelope),
        "risk_score": risk_payload.get("risk_score"),
        "risk_level": risk_payload.get("risk_level"),
        "confidence": confidence_of(risk_envelope),
        "breakdown": risk_payload.get("confidence_breakdown"),
    }

    # ── Prioritization ──
    prio = state.get("prioritization") or {}
    signals["prioritization"] = {
        "priority_level": prio.get("priority_level"),
        "recommended_focus": prio.get("recommended_focus"),
        "confidence_score": prio.get("confidence_score"),
        "growth_short": prio.get("growth_short"),
        "growth_mid": prio.get("growth_mid"),
        "growth_long": prio.get("growth_long"),
        "growth_trend_acceleration": prio.get("growth_trend_acceleration"),
        "cohort_signal_used": prio.get("cohort_signal_used"),
        "cohort_risk_hint": prio.get("cohort_risk_hint"),
        "scenario_signal_used": prio.get("scenario_signal_used"),
        "scenario_worst_growth": prio.get("scenario_worst_growth"),
        "scenario_best_growth": prio.get("scenario_best_growth"),
        "signal_conflict_count": prio.get("signal_conflict_count"),
        "signal_conflict_warnings": prio.get("signal_conflict_warnings"),
    }

    # ── Growth horizons ──
    growth_envelope = state.get("growth_data")
    growth_payload = payload_of(growth_envelope) or {}
    signals["growth"] = {
        "status": status_of(growth_envelope),
        "confidence": confidence_of(growth_envelope),
        "short_growth": growth_payload.get("short_growth"),
        "mid_growth": growth_payload.get("mid_growth"),
        "long_growth": growth_payload.get("long_growth"),
        "trend_acceleration": growth_payload.get("trend_acceleration"),
        "metric_series": growth_payload.get("metric_series"),
    }

    # ── Forecast ──
    forecast_envelope = state.get("forecast_data")
    forecast_payload = payload_of(forecast_envelope) or {}
    signals["forecast"] = {
        "status": status_of(forecast_envelope),
        "confidence": confidence_of(forecast_envelope),
        "forecasts": forecast_payload.get("forecasts"),
        "metrics_queried": forecast_payload.get("metrics_queried"),
    }

    # ── Cohort analytics ──
    cohort_envelope = state.get("cohort_data")
    cohort_payload = payload_of(cohort_envelope) or {}
    signals["cohort"] = {
        "status": status_of(cohort_envelope),
        "confidence": confidence_of(cohort_envelope),
        "retention_decay": cohort_payload.get("retention_decay"),
        "churn_acceleration": cohort_payload.get("churn_acceleration"),
        "worst_cohort": cohort_payload.get("worst_cohort"),
        "risk_hint": cohort_payload.get("risk_hint"),
    }

    # ── Signal integrity ──
    integrity = state.get("signal_integrity")
    if isinstance(integrity, dict):
        signals["signal_integrity"] = integrity

    # ── Signal conflicts ──
    conflict_envelope = state.get("signal_conflicts")
    conflict_payload = payload_of(conflict_envelope) or {}
    # The conflict_result dict is nested inside the payload under
    # "conflict_result" — extract fields from there.
    conflict_result = conflict_payload.get("conflict_result") or {}
    signals["signal_conflicts"] = {
        "status": status_of(conflict_envelope),
        "conflict_count": conflict_result.get("conflict_count", 0),
        "conflicts": conflict_result.get("conflicts"),
        "total_severity": conflict_result.get("total_severity"),
        "uncertainty_flag": conflict_result.get("uncertainty_flag", False),
        "warnings": conflict_result.get("warnings"),
    }

    # ── Unit economics ──
    unit_envelope = state.get("unit_economics_data")
    unit_payload = payload_of(unit_envelope) or {}
    signals["unit_economics"] = {
        "status": status_of(unit_envelope),
        "confidence": confidence_of(unit_envelope),
        "ltv": unit_payload.get("ltv"),
        "cac": unit_payload.get("cac"),
        "ltv_cac_ratio": unit_payload.get("ltv_cac_ratio"),
        "payback_months": unit_payload.get("payback_months"),
    }

    # ── Multivariate scenarios ──
    scenario_envelope = state.get("multivariate_scenario_data")
    scenario_payload = payload_of(scenario_envelope) or {}
    signals["scenarios"] = {
        "status": status_of(scenario_envelope),
        "confidence": confidence_of(scenario_envelope),
        "scenario_simulation": scenario_payload.get("scenario_simulation"),
    }

    # ── Competitive benchmark ──
    benchmark_envelope = state.get("benchmark_data")
    benchmark_payload = payload_of(benchmark_envelope) or {}
    if benchmark_payload:
        signals["benchmark"] = {
            "status": status_of(benchmark_envelope),
            "confidence": confidence_of(benchmark_envelope),
            "ranking": benchmark_payload.get("ranking"),
            "composite": benchmark_payload.get("composite"),
            "peer_selection": benchmark_payload.get("peer_selection"),
            "market_position": benchmark_payload.get("market_position"),
            "metric_comparison_specs": benchmark_payload.get("metric_comparison_specs"),
        }

    # ── Pipeline status ──
    signals["pipeline_status"] = state.get("pipeline_status", "partial")
    signals["dataset_confidence"] = state.get("dataset_confidence")
    signals["synthesis_blocked"] = state.get("synthesis_blocked")

    return signals


@router.post("/analyze")
def analyze(
    response: Response,
    prompt: str = Form(..., description="Business prompt / user query"),
    file: Optional[UploadFile] = File(default=None, description="Optional CSV file"),
    client_id: Optional[str] = Form(default=None),
    business_type: Optional[str] = Form(default=None),
    multi_entity_behavior: Optional[str] = Form(default=None),
    competitors: Optional[str] = Form(default=None, description="Comma-separated competitor names for web-based benchmarking"),
    self_analysis_only: Optional[bool] = Form(default=None, description="When true, force data-only self analysis and disable competitor benchmarking in synthesis."),
    model: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
    csv_service: CSVIngestionService = Depends(get_csv_ingestion_service),
    security: SecurityContext = Depends(require_security_context),
) -> dict[str, Any]:
    """Run the full insight generation pipeline.

    Accepts an optional CSV file and query parameters. Internally:
    1. Extract intent from prompt
    2. Ingest CSV if present (no analytics trigger — entity unknown yet)
    3. Resolve entity_name from dataset records (dataset-first, prompt-fallback)
    4. Ensure KPI + forecast data exist for the resolved entity
    5. Invoke LangGraph pipeline
    6. Guard against empty KPI data
    7. Return structured InsightOutput
    """
    model_name = str(model or "").strip()
    llm_model_override: str | None = model_name if model_name and model_name != "default" else None

    # Initialise outside try so catch blocks can always reference them.
    pipeline_signals: dict[str, Any] = {}
    state: dict[str, Any] = {}

    try:
        # Step 1: Resolve business_type only (entity_name is never inferred from prompt).
        seed_state = intent_node({"user_query": prompt})
        resolved_business_type = seed_state.get("business_type")
        requested_entity = client_id.strip() if client_id and client_id.strip() else None
        if requested_entity:
            assert_entity_allowed_for_tenant(entity_name=requested_entity, security=security)

        # Override with explicit params if provided
        if business_type:
            resolved_business_type = business_type

        processing_strategy = get_processing_strategy(resolved_business_type)
        supported = set(supported_categories())
        analytics_strategy = (
            processing_strategy
            if processing_strategy in supported
            else None
        )

        # Step 2: CSV ingestion.
        dataset_uploaded = file is not None
        ingestion_started_at = datetime.now(tz=timezone.utc) - timedelta(seconds=1)
        fallback_upload_entities: list[str] = []
        ingestion_confidence = 1.0
        ingestion_warnings: list[str] = []
        ingestion_provenance: dict[str, Any] = {}
        ingestion_pipeline_status: str | None = None
        ingestion_inferred_category: str | None = None
        ingestion_category_confidence: float = 0.0
        if dataset_uploaded:
            try:
                effective_multi_entity_behavior = multi_entity_behavior or (
                    "split" if requested_entity else None
                )
                ingest_summary = csv_service.ingest_csv(
                    upload_file=file,
                    db=db,
                    client_name=None,
                    tenant_id=security.tenant_id,
                    multi_entity_behavior=effective_multi_entity_behavior,
                )
                if ingest_summary.validation_errors:
                    fatal_errors = [
                        err
                        for err in ingest_summary.validation_errors
                        if str(err.code or "").strip().lower() not in {"schema_interpreter_warning"}
                    ]
                    ingestion_status_str = str(
                        ingest_summary.pipeline_status or ""
                    ).strip().lower()
                    is_hard_failure = ingestion_status_str in (
                        IngestionStatus.FAILED.value,
                        IngestionStatus.SCHEMA_MISMATCH.value,
                    )
                    if _is_benign_no_valid_records(ingest_summary):
                        try:
                            fallback_upload_entities = csv_service.detect_csv_entities(
                                upload_file=file,
                                db=db,
                                client_name=None,
                            )
                        except Exception:
                            logger.debug(
                                "Analyze: could not detect fallback upload entities.",
                                exc_info=True,
                            )
                        logger.info(
                            "Analyze: ingestion produced no new rows (likely deduped). "
                            "rows_failed=%s fallback_upload_entities=%s",
                            ingest_summary.rows_failed,
                            fallback_upload_entities,
                        )
                    elif is_hard_failure and fatal_errors and ingest_summary.rows_processed == 0:
                        first_error = fatal_errors[0]
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=build_error_detail(
                                code=INGESTION_VALIDATION,
                                message=first_error.message,
                                error_type="ingestion_validation_failed",
                                context={
                                    "validation_code": first_error.code,
                                    "column": first_error.column,
                                    "context": first_error.context,
                                    "ingestion_status": ingestion_status_str,
                                },
                            ),
                        )
                    elif fatal_errors and ingest_summary.rows_processed == 0:
                        # Soft failure (insufficient_data / empty_dataset):
                        # log but do NOT abort — existing DB data may suffice.
                        logger.warning(
                            "Analyze: ingestion status=%s with zero rows processed. "
                            "Attempting to continue with existing DB data. "
                            "rows_failed=%s",
                            ingestion_status_str,
                            ingest_summary.rows_failed,
                        )
                    elif fatal_errors:
                        logger.warning(
                            "Analyze: continuing with partial ingestion rows_processed=%s rows_failed=%s",
                            ingest_summary.rows_processed,
                            ingest_summary.rows_failed,
                        )
                ingestion_confidence = float(ingest_summary.confidence_score or 1.0)
                ingestion_confidence = max(0.0, min(1.0, ingestion_confidence))
                ingestion_warnings = [str(item) for item in (ingest_summary.warnings or [])]
                ingestion_provenance = (
                    ingest_summary.provenance
                    if isinstance(ingest_summary.provenance, dict)
                    else {}
                )
                ingestion_pipeline_status = str(ingest_summary.pipeline_status or "").strip() or None
                ingestion_inferred_category = ingest_summary.inferred_category
                ingestion_category_confidence = float(
                    ingest_summary.category_confidence or 0.0
                )
            finally:
                file.file.close()

        # Step 3: Entity resolution — select ONE target, treat rest as peers.
        # 1) If client_id provided → use it as target.
        # 2) Else if dataset uploaded → pick target deterministically.
        # 3) Else → structured validation error.
        peer_entities: list[str] = []
        if requested_entity:
            resolved_entity_name = requested_entity
        elif dataset_uploaded:
            dataset_entities = _extract_distinct_entity_names_from_canonical_records(
                db=db,
                created_since=ingestion_started_at,
            )
            used_fallback_entities = False
            if not dataset_entities and fallback_upload_entities:
                used_fallback_entities = True
                dataset_entities = sorted(
                    {
                        str(entity).strip()
                        for entity in fallback_upload_entities
                        if str(entity).strip()
                    }
                )
            if not dataset_entities:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=build_error_detail(
                        code=SCHEMA_CONFLICT,
                        message="No entity_name values were found in ingested canonical records.",
                        error_type="entity_resolution_failed",
                    ),
                )
            if len(dataset_entities) == 1:
                resolved_entity_name = dataset_entities[0]
            else:
                # Multi-entity dataset: pick the entity with the most records
                # as the analysis target; the rest become benchmark peers.
                # When rows were deduped (fallback path), count ALL records
                # instead of only recently created ones.
                entity_counts = _count_records_per_entity(
                    db=db,
                    entities=dataset_entities,
                    created_since=None if used_fallback_entities else ingestion_started_at,
                )
                resolved_entity_name = max(
                    dataset_entities, key=lambda e: entity_counts.get(e, 0),
                )
                peer_entities = [e for e in dataset_entities if e != resolved_entity_name]
                logger.info(
                    "Multi-entity dataset: resolved_entity=%r peer_entities=%r entity_count=%d",
                    resolved_entity_name,
                    peer_entities,
                    len(dataset_entities),
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=build_error_detail(
                    code=SCHEMA_CONFLICT,
                    message="client_id is required when no dataset is uploaded.",
                    error_type="entity_resolution_failed",
                ),
            )
        # If client_id was provided explicitly, detect peers from DB.
        if requested_entity and dataset_uploaded and not peer_entities:
            dataset_entities = _extract_distinct_entity_names_from_canonical_records(
                db=db, created_since=ingestion_started_at,
            )
            peer_entities = [e for e in dataset_entities if e != resolved_entity_name]
        assert_entity_allowed_for_tenant(entity_name=resolved_entity_name, security=security)

        if response is not None:
            response.headers["X-Resolved-Entity-Name"] = resolved_entity_name
            response.headers["X-Tenant-ID"] = security.tenant_id

        if dataset_uploaded and not business_type:
            # Dataset category is authoritative over prompt-inferred business
            # type.  Reset analytics_strategy so the dataset gets a chance to
            # override whatever the intent node guessed from the prompt.
            dataset_inferred_strategy: str | None = None

            # Prefer category inference from ingestion (confidence-scored)
            if (
                ingestion_inferred_category
                and ingestion_category_confidence >= 0.80
            ):
                inferred_pack_strategy = get_processing_strategy(ingestion_inferred_category)
                if inferred_pack_strategy and inferred_pack_strategy in supported:
                    dataset_inferred_strategy = inferred_pack_strategy
                    logger.info(
                        "Analyze: inferred processing_strategy=%r from category inference "
                        "(confidence=%.2f) for entity=%r",
                        inferred_pack_strategy,
                        ingestion_category_confidence,
                        resolved_entity_name,
                    )

            # Fallback to dataset category scan
            if not dataset_inferred_strategy:
                inferred_strategy = _infer_analytics_strategy_from_dataset(
                    db=db,
                    entity_name=resolved_entity_name,
                )
                if inferred_strategy:
                    dataset_inferred_strategy = inferred_strategy
                    logger.info(
                        "Analyze: inferred processing_strategy=%r from dataset categories "
                        "for entity=%r",
                        inferred_strategy,
                        resolved_entity_name,
                    )

            if dataset_inferred_strategy:
                processing_strategy = dataset_inferred_strategy
                analytics_strategy = dataset_inferred_strategy

        # Step 4: Ensure KPI + forecast data exist for the resolved entity.
        # On first upload the computed_kpis table is empty for this entity;
        # the graph's KPI-fetch nodes only *query* — they don't compute.
        if resolved_entity_name and analytics_strategy:
            if response is not None:
                response.headers["X-Resolved-Business-Type"] = analytics_strategy
            _ensure_analytics_data(
                entity_name=resolved_entity_name,
                processing_strategy=analytics_strategy,
                db=db,
            )

        if response is not None and processing_strategy:
            response.headers["X-Resolved-Business-Type"] = processing_strategy

        # Early guard: if we still have no business type, fail gracefully.
        if not processing_strategy:
            supported = ", ".join(f"'{value}'" for value in supported_categories())
            return InsightOutput.failure(
                "Could not determine category from prompt. "
                f"Specify business_type/category as one of: {supported}."
            ).model_dump()

        # Pre-flight: verify KPI data exists before invoking the graph.
        # The graph's risk_node calls normalize_signals() which raises
        # ValueError on empty KPI records — catching it here gives a
        # clearer message than the signal-normalizer stack trace.
        if not _has_any_computed_kpis(db, resolved_entity_name):
            return InsightOutput.failure(
                f"No KPI records found for entity '{resolved_entity_name}'. "
                "Ensure the uploaded CSV contains valid metric data "
                "(entity_name, metric_name, metric_value, timestamp)."
            ).model_dump()

        # Step 5: Invoke graph
        invoke_state: dict[str, Any] = {
            "request_id": security.request_id,
            "user_query": prompt,
            "business_type": processing_strategy,
            "entity_name": resolved_entity_name,
            "peer_entities": peer_entities,
        }
        if peer_entities:
            logger.info(
                "Graph invoke: entity=%r peers=%r (%d peers)",
                resolved_entity_name, peer_entities, len(peer_entities),
            )
        dataset_confidence = (
            ingestion_confidence
            if dataset_uploaded
            else _estimate_dataset_confidence(
                db=db,
                entity_name=resolved_entity_name,
                processing_strategy=processing_strategy,
            )
        )
        invoke_state["dataset_confidence"] = dataset_confidence
        if llm_model_override:
            invoke_state["llm_model_override"] = llm_model_override
        if ingestion_warnings:
            invoke_state["ingestion_warnings"] = ingestion_warnings
        if ingestion_provenance:
            invoke_state["ingestion_provenance"] = ingestion_provenance
        # Self-analysis mode unless peers are available from multi-entity dataset.
        # When peers exist, benchmark_node can run competitive analysis locally.
        invoke_state["self_analysis_only"] = len(peer_entities) == 0

        graph_timeout = float(os.getenv("GRAPH_INVOKE_TIMEOUT_SECONDS", "180"))
        import time as _time
        _graph_start = _time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(insight_graph.invoke, invoke_state)
                state = future.result(timeout=graph_timeout)
        except FuturesTimeoutError:
            _elapsed = _time.perf_counter() - _graph_start
            logger.error(
                "Analyze: graph invocation timed out after %.1fs (limit=%.0fs) entity=%r",
                _elapsed,
                graph_timeout,
                resolved_entity_name,
            )
            result = InsightOutput.failure(
                f"Analysis timed out after {int(graph_timeout)}s. "
                "Try a smaller dataset or simpler query."
            ).model_dump()
            # Try to extract any pipeline signals from partial state.
            if state:
                pipeline_signals = _extract_pipeline_signals(state)
            result["pipeline_signals"] = pipeline_signals
            return result

        _elapsed = _time.perf_counter() - _graph_start
        logger.info("Analyze: graph completed in %.1fs entity=%r", _elapsed, resolved_entity_name)
        integrity_payload = state.get("signal_integrity")
        if not isinstance(integrity_payload, dict):
            integrity_payload = UnifiedSignalIntegrity.compute(state)
        integrity_scores = UnifiedSignalIntegrity.score_vector_from_integrity(integrity_payload)
        logger.info("Analyze signal integrity: %s", json.dumps(integrity_scores, sort_keys=True))

        # Extract pipeline signals for the frontend (always, even on failure).
        pipeline_signals = _extract_pipeline_signals(state)

        # Step 6: Guard — empty KPI records (post-graph safety net)
        if not _has_kpi_data(state):
            failure_output = InsightOutput.failure(
                f"No KPI records found for entity '{resolved_entity_name}'. "
                "Upload a dataset or verify that historical data exists."
            )
            result = failure_output.model_dump()
            result["pipeline_signals"] = pipeline_signals
            return result

        # Step 7: Extract and validate output
        final_response_json = state.get("final_response")
        if not isinstance(final_response_json, str):
            # No final_response: synthesis may have been blocked or LLM failed.
            # Return a structured failure but KEEP pipeline_signals so the
            # frontend can still render charts, trends, and diagnostics.
            failure_output = InsightOutput.failure(
                "Pipeline did not produce final_response."
            )
            result = failure_output.model_dump()
            result["pipeline_signals"] = pipeline_signals
            envelope_diag = state.get("envelope_diagnostics")
            if isinstance(envelope_diag, dict):
                result["diagnostics"] = {
                    "warnings": envelope_diag.get("warnings", []),
                    "confidence_score": envelope_diag.get("confidence_score"),
                    "missing_signal": envelope_diag.get("missing_signal", []),
                    "confidence_adjustments": envelope_diag.get(
                        "confidence_adjustments", []
                    ),
                }
            return result

        payload = json.loads(final_response_json)
        output_model = InsightOutput.model_validate(payload)

        result = output_model.model_dump()
        result["pipeline_signals"] = pipeline_signals

        # Step 8: Reconcile top-level pipeline_status with signal integrity.
        # The InsightOutput.pipeline_status property derives from confidence,
        # but the authoritative status comes from signal_integrity.insight_quality.
        _insight_quality = str(
            (integrity_payload or {}).get("insight_quality", "")
        ).lower()
        _quality_to_status = {
            "full_insight": "success",
            "partial_insight": "partial",
            "blocked": "failed",
        }
        if _insight_quality in _quality_to_status:
            result["pipeline_status"] = _quality_to_status[_insight_quality]

        # Expose insight mode as a top-level flag so the frontend can
        # adjust UX without parsing prefixes from every text field.
        result["mode"] = _insight_quality or "full_insight"

        # Step 9: Merge envelope_diagnostics into response (generated by
        # synthesis_gate / llm_node but not part of InsightOutput schema).
        envelope_diag = state.get("envelope_diagnostics")
        if isinstance(envelope_diag, dict):
            result["diagnostics"] = {
                "warnings": envelope_diag.get("warnings", []),
                "confidence_score": envelope_diag.get("confidence_score"),
                "missing_signal": envelope_diag.get("missing_signal", []),
                "confidence_adjustments": envelope_diag.get(
                    "confidence_adjustments", []
                ),
            }

        return result

    except HTTPException:
        raise
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        logger.exception("Analysis pipeline failed")
        failure_output = InsightOutput.failure(str(exc))
        result = failure_output.model_dump()
        # Preserve pipeline_signals extracted before the failure — they
        # contain chart data (trends, growth, forecasts) the frontend needs.
        result["pipeline_signals"] = pipeline_signals
        envelope_diag = state.get("envelope_diagnostics")
        if isinstance(envelope_diag, dict):
            result["diagnostics"] = {
                "warnings": envelope_diag.get("warnings", []),
                "confidence_score": envelope_diag.get("confidence_score"),
                "missing_signal": envelope_diag.get("missing_signal", []),
                "confidence_adjustments": envelope_diag.get(
                    "confidence_adjustments", []
                ),
            }
        return result
    except Exception as exc:
        logger.exception("Analysis pipeline unexpected error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(
                code=INTERNAL_FAILURE,
                message=f"Pipeline error: {exc}",
            ),
        ) from exc


