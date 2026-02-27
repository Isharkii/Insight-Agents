"""
app/api/routers/analyze_router.py

Insight generation endpoint — exposes the full LangGraph pipeline as an API.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agent.graph import insight_graph
from agent.nodes.intent import intent_node
from app.failure_codes import (
    INGESTION_VALIDATION,
    INTERNAL_FAILURE,
    SCHEMA_CONFLICT,
    build_error_detail,
)
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
from app.services.kpi_orchestrator import KPIOrchestrator, KPIRunResult
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


def _has_computed_kpis(db: Session, entity_name: str) -> bool:
    """Check whether ComputedKPI rows exist for *entity_name*."""
    repo = KPIRepository(db)
    rows = repo.get_kpis_by_period(
        period_start=datetime(1970, 1, 1, tzinfo=timezone.utc),
        period_end=datetime.now(tz=timezone.utc),
        entity_name=entity_name,
    )
    return bool(rows)


def _ensure_analytics_data(
    *,
    entity_name: str,
    processing_strategy: str,
    db: Session,
) -> None:
    """Run KPI computation + forecast for *entity_name* if not already present.

    Mirrors the analytics trigger in CSVIngestionService but uses the
    *correct* entity_name extracted from canonical records.
    """
    if _has_computed_kpis(db, entity_name):
        return

    period_start, now = _resolve_kpi_period_from_entity_data(
        db=db,
        entity_name=entity_name,
        processing_strategy=processing_strategy,
    )

    # --- KPI computation ---
    kpi_result: KPIRunResult | None = None
    try:
        kpi_result = KPIOrchestrator().run(
            entity_name=entity_name,
            business_type=processing_strategy,
            period_start=period_start,
            period_end=now,
            db=db,
        )
        logger.info(
            "Analyze: KPI computed entity=%r processing_strategy=%r record_id=%s",
            entity_name,
            processing_strategy,
            kpi_result.record_id,
        )
    except Exception:
        logger.warning(
            "Analyze: KPI computation failed entity=%r processing_strategy=%r",
            entity_name,
            processing_strategy,
            exc_info=True,
        )

    # --- Forecast generation ---
    try:
        metric_name = primary_metric_for_business_type(processing_strategy)
        values = _extract_primary_metric_values(kpi_result, metric_name)
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
                "Analyze: forecast generated entity=%r metric=%r",
                entity_name,
                metric_name,
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

    Falls back to [now-90d, now] when no timestamps are available.
    """
    if latest is None:
        return now - timedelta(days=90), now

    latest_utc = _ensure_utc(latest)
    candidate_start = latest_utc - timedelta(days=90)
    if earliest is None:
        return candidate_start, latest_utc

    earliest_utc = _ensure_utc(earliest)
    return max(earliest_utc, candidate_start), latest_utc


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
    """Check whether the pipeline produced any KPI records."""
    for key, payload in state.items():
        if key != "kpi_data" and not key.endswith("_kpi_data"):
            continue
        if isinstance(payload, dict):
            data = payload.get("payload") if "status" in payload else payload
            records = data.get("records") if isinstance(data, dict) else None
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


@router.post("/analyze", response_model=InsightOutput)
def analyze(
    prompt: str = Form(..., description="Business prompt / user query"),
    file: Optional[UploadFile] = File(default=None, description="Optional CSV file"),
    client_id: Optional[str] = Form(default=None),
    business_type: Optional[str] = Form(default=None),
    multi_entity_behavior: Optional[str] = Form(default=None),
    model: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
    csv_service: CSVIngestionService = Depends(get_csv_ingestion_service),
) -> InsightOutput:
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
    if model and model != "default":
        os.environ["LLM_MODEL"] = model

    try:
        # Step 1: Resolve business_type only (entity_name is never inferred from prompt).
        seed_state = intent_node({"user_query": prompt})
        resolved_business_type = seed_state.get("business_type")
        requested_entity = client_id.strip() if client_id and client_id.strip() else None

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

        # Step 3: Strict entity_name resolution order.
        # 1) If client_id provided -> use it.
        # 2) Else if dataset uploaded -> extract distinct entity_name from canonical records.
        # 3) Else -> structured validation error.
        if requested_entity:
            resolved_entity_name = requested_entity
        elif dataset_uploaded:
            dataset_entities = _extract_distinct_entity_names_from_canonical_records(
                db=db,
                created_since=ingestion_started_at,
            )
            if not dataset_entities and fallback_upload_entities:
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
            if len(dataset_entities) > 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=build_error_detail(
                        code=SCHEMA_CONFLICT,
                        message="Multiple entity_name values detected in dataset.",
                        error_type="entity_resolution_failed",
                        context={"entity_names": dataset_entities},
                    ),
                )
            resolved_entity_name = dataset_entities[0]
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=build_error_detail(
                    code=SCHEMA_CONFLICT,
                    message="client_id is required when no dataset is uploaded.",
                    error_type="entity_resolution_failed",
                ),
            )

        if dataset_uploaded and not analytics_strategy and not business_type:
            # Prefer category inference from ingestion (confidence-scored)
            if (
                ingestion_inferred_category
                and ingestion_category_confidence >= 0.80
            ):
                inferred_pack_strategy = get_processing_strategy(ingestion_inferred_category)
                if inferred_pack_strategy and inferred_pack_strategy in supported:
                    processing_strategy = inferred_pack_strategy
                    analytics_strategy = inferred_pack_strategy
                    logger.info(
                        "Analyze: inferred processing_strategy=%r from category inference "
                        "(confidence=%.2f) for entity=%r",
                        inferred_pack_strategy,
                        ingestion_category_confidence,
                        resolved_entity_name,
                    )

            # Fallback to dataset category scan
            if not analytics_strategy:
                inferred_strategy = _infer_analytics_strategy_from_dataset(
                    db=db,
                    entity_name=resolved_entity_name,
                )
                if inferred_strategy:
                    processing_strategy = inferred_strategy
                    analytics_strategy = inferred_strategy
                    logger.info(
                        "Analyze: inferred processing_strategy=%r from dataset categories "
                        "for entity=%r",
                        inferred_strategy,
                        resolved_entity_name,
                    )

        # Step 4: Ensure KPI + forecast data exist for the resolved entity.
        # On first upload the computed_kpis table is empty for this entity;
        # the graph's KPI-fetch nodes only *query* — they don't compute.
        if resolved_entity_name and analytics_strategy:
            _ensure_analytics_data(
                entity_name=resolved_entity_name,
                processing_strategy=analytics_strategy,
                db=db,
            )

        # Early guard: if we still have no business type, fail gracefully.
        if not processing_strategy:
            supported = ", ".join(f"'{value}'" for value in supported_categories())
            return InsightOutput.failure(
                "Could not determine category from prompt. "
                f"Specify business_type/category as one of: {supported}."
            )

        # Pre-flight: verify KPI data exists before invoking the graph.
        # The graph's risk_node calls normalize_signals() which raises
        # ValueError on empty KPI records — catching it here gives a
        # clearer message than the signal-normalizer stack trace.
        if not _has_computed_kpis(db, resolved_entity_name):
            return InsightOutput.failure(
                f"No KPI records found for entity '{resolved_entity_name}'. "
                "Ensure the uploaded CSV contains valid metric data "
                "(entity_name, metric_name, metric_value, timestamp)."
            )

        # Step 5: Invoke graph
        invoke_state: dict[str, Any] = {
            "user_query": prompt,
            "business_type": processing_strategy,
            "entity_name": resolved_entity_name,
        }
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
        if ingestion_warnings:
            invoke_state["ingestion_warnings"] = ingestion_warnings
        if ingestion_provenance:
            invoke_state["ingestion_provenance"] = ingestion_provenance

        state = insight_graph.invoke(invoke_state)

        # Step 6: Guard — empty KPI records (post-graph safety net)
        if not _has_kpi_data(state):
            return InsightOutput.failure(
                f"No KPI records found for entity '{resolved_entity_name}'. "
                "Upload a dataset or verify that historical data exists."
            )

        # Step 7: Extract and validate output
        response = state.get("final_response")
        if not isinstance(response, str):
            raise ValueError("Pipeline did not produce final_response.")

        payload = json.loads(response)
        output_model = InsightOutput.model_validate(payload)

        # Keep strict response schema unchanged while surfacing ingestion
        # pipeline state as diagnostics warnings.
        diagnostics = output_model.diagnostics
        if ingestion_pipeline_status and diagnostics is not None:
            if ingestion_pipeline_status not in (
                IngestionStatus.SUCCESS.value,
                "success",
            ):
                warnings = list(diagnostics.warnings)
                # Differentiate messaging by ingestion status severity.
                if ingestion_pipeline_status in (
                    IngestionStatus.FAILED.value,
                    IngestionStatus.SCHEMA_MISMATCH.value,
                ):
                    warnings.append(
                        f"Ingestion pipeline status was {ingestion_pipeline_status}; "
                        f"data could not be loaded from the uploaded file."
                    )
                elif ingestion_pipeline_status == IngestionStatus.EMPTY_DATASET.value:
                    warnings.append(
                        f"Ingestion pipeline status was {ingestion_pipeline_status}; "
                        f"the uploaded file contained no usable data rows."
                    )
                elif ingestion_pipeline_status == IngestionStatus.INSUFFICIENT_DATA.value:
                    warnings.append(
                        f"Ingestion pipeline status was {ingestion_pipeline_status}; "
                        f"partial data path applied — some rows failed validation."
                    )
                else:
                    warnings.append(
                        f"Ingestion pipeline status was {ingestion_pipeline_status}; "
                        f"partial data path applied."
                    )
                output_model = output_model.model_copy(
                    update={
                        "diagnostics": diagnostics.model_copy(update={"warnings": warnings})
                    }
                )
        return output_model

    except HTTPException:
        raise
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        logger.exception("Analysis pipeline failed")
        return InsightOutput.failure(str(exc))
    except Exception as exc:
        logger.exception("Analysis pipeline unexpected error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(
                code=INTERNAL_FAILURE,
                message=f"Pipeline error: {exc}",
            ),
        ) from exc
