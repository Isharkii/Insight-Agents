"""
app/api/routers/kpi_router.py

KPI recompute endpoint.

Triggers the full analytics pipeline for a given entity:
    KPIOrchestrator → ForecastOrchestrator → RiskOrchestrator → SegmentationOrchestrator

Each step is independent; failures are collected and returned in the response
without aborting subsequent steps or returning a non-2xx status.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.services.csv_ingestion_service import (
    _PRIMARY_METRIC_BY_BUSINESS_TYPE,
    _build_segmentation_records,
    _extract_primary_metric_values,
)
from app.services.kpi_orchestrator import (
    KPIOrchestrator,
    KPIRunResult,
    KPIUnknownBusinessTypeError,
    KPIAggregationError,
    KPIPersistenceError,
)
from db.session import get_db
from forecast.orchestrator import ForecastOrchestrator
from risk.orchestrator import RiskOrchestrator
from segmentation.orchestrator import SegmentationOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["kpi"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class KPIRecomputeRequest(BaseModel):
    entity_name: str
    business_type: str


class KPIRecomputeResponse(BaseModel):
    entity_name: str
    business_type: str
    kpi: dict | None = None
    forecast: dict | None = None
    risk: dict | None = None
    segmentation: dict | None = None
    pipeline_errors: list[str] = []


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/recompute-kpis",
    response_model=KPIRecomputeResponse,
    status_code=status.HTTP_200_OK,
)
def recompute_kpis(
    body: KPIRecomputeRequest,
    db: Session = Depends(get_db),
) -> KPIRecomputeResponse:
    """
    Trigger the full analytics pipeline for one entity.

    All four orchestrators run sequentially. A failure in any step is
    captured in ``pipeline_errors`` and does not prevent subsequent steps
    from executing. The response always returns HTTP 200 when the request
    itself is valid; per-step failures are surfaced inside the payload.

    Raises HTTP 400 for an unrecognised ``business_type``.
    Raises HTTP 500 for unrecoverable aggregation or persistence failures.
    """
    now = datetime.now(tz=timezone.utc)
    period_start = now - timedelta(days=90)

    pipeline_errors: list[str] = []
    kpi_result: KPIRunResult | None = None
    kpi_summary: dict | None = None
    forecast_summary: dict | None = None
    risk_summary: dict | None = None
    seg_summary: dict | None = None

    # --- Step 1: KPI recomputation ---
    try:
        kpi_result = KPIOrchestrator().run(
            entity_name=body.entity_name,
            business_type=body.business_type,
            period_start=period_start,
            period_end=now,
            db=db,
        )
        kpi_summary = {
            "record_id": str(kpi_result.record_id),
            "has_errors": kpi_result.has_errors,
            "metrics": kpi_result.metrics,
        }
        logger.info(
            "KPI recomputed entity=%r business_type=%r record_id=%s",
            body.entity_name,
            body.business_type,
            kpi_result.record_id,
        )
    except KPIUnknownBusinessTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported business_type: {body.business_type}",
        ) from exc
    except (KPIAggregationError, KPIPersistenceError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"KPI pipeline failed: {exc}",
        ) from exc

    # --- Step 2: Forecast ---
    try:
        metric_name = _PRIMARY_METRIC_BY_BUSINESS_TYPE.get(body.business_type, "revenue")
        values = _extract_primary_metric_values(kpi_result, metric_name)
        forecast_result = ForecastOrchestrator(db).generate_forecast(
            entity_name=body.entity_name,
            metric_name=metric_name,
            values=values,
        )
        if "error" in forecast_result:
            pipeline_errors.append(f"forecast: {forecast_result['error']}")
        else:
            db.commit()
            forecast_summary = forecast_result
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        pipeline_errors.append(f"forecast: {exc}")
        logger.warning("Forecast failed entity=%r: %s", body.entity_name, exc)

    # --- Step 3: Risk scoring ---
    try:
        risk_result = RiskOrchestrator(db).generate_risk_score(
            entity_name=body.entity_name,
            kpi_data={},
            forecast_data=forecast_summary or {},
        )
        db.commit()
        risk_summary = risk_result
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        pipeline_errors.append(f"risk: {exc}")
        logger.warning("Risk scoring failed entity=%r: %s", body.entity_name, exc)

    # --- Step 4: Segmentation ---
    try:
        seg_records = _build_segmentation_records(kpi_result)
        n_clusters = min(3, len(seg_records))
        if n_clusters < 1:
            pipeline_errors.append("segmentation: insufficient records for clustering")
        else:
            seg_result = SegmentationOrchestrator(session=db).run_segmentation(
                entity_name=body.entity_name,
                records=seg_records,
                n_clusters=n_clusters,
            )
            db.commit()
            seg_summary = {"n_clusters": seg_result.get("n_clusters")}
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        pipeline_errors.append(f"segmentation: {exc}")
        logger.warning("Segmentation failed entity=%r: %s", body.entity_name, exc)

    return KPIRecomputeResponse(
        entity_name=body.entity_name,
        business_type=body.business_type,
        kpi=kpi_summary,
        forecast=forecast_summary,
        risk=risk_summary,
        segmentation=seg_summary,
        pipeline_errors=pipeline_errors,
    )
