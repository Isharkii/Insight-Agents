"""
app/api/routers/kpi_router.py

KPI recompute endpoint.

Triggers the full analytics pipeline for a given entity:
    KPIOrchestrator → ForecastOrchestrator → RiskOrchestrator

Segmentation is optional and must be explicitly requested.

Each step is independent; failures are collected and returned in the response
without aborting subsequent steps or returning a non-2xx status.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from dateutil.relativedelta import relativedelta
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.failure_codes import INTERNAL_FAILURE, SCHEMA_CONFLICT, build_error_detail
from app.security.dependencies import (
    assert_entity_allowed_for_tenant,
    require_security_context,
)
from app.security.models import SecurityContext
from app.services.category_registry import primary_metric_for_business_type
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

logger = logging.getLogger(__name__)

router = APIRouter(tags=["kpi"])

# ---------------------------------------------------------------------------
# Analytics helpers (formerly in csv_ingestion_service)
# ---------------------------------------------------------------------------


def _extract_primary_metric_values(
    kpi_result: KPIRunResult | None,
    metric_name: str,
) -> list[float]:
    """Return a single-element list from the named KPI metric value."""
    if kpi_result is None:
        return []
    entry = kpi_result.metrics.get(metric_name, {})
    value = entry.get("value")
    if not isinstance(value, (int, float)):
        return []
    return [float(value)]


def _generate_monthly_windows(
    period_start: datetime,
    period_end: datetime,
) -> list[tuple[datetime, datetime]]:
    """Split a date range into calendar-month windows."""
    start = period_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    windows: list[tuple[datetime, datetime]] = []
    while start <= period_end:
        end = start + relativedelta(months=1)
        windows.append((start, end))
        start = end
    return windows


def _build_segmentation_records(
    kpi_result: KPIRunResult | None,
) -> list[dict]:
    """Build a flat metric record list from a KPI result for segmentation."""
    if kpi_result is None:
        return []
    flat: dict = {}
    for name, entry in kpi_result.metrics.items():
        value = entry.get("value")
        if isinstance(value, (int, float)):
            flat[name] = float(value)
    return [flat] if flat else []


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class KPIRecomputeRequest(BaseModel):
    entity_name: str
    business_type: str
    include_segmentation: bool = False


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
    security: SecurityContext = Depends(require_security_context),
) -> KPIRecomputeResponse:
    """
    Trigger the full analytics pipeline for one entity.

    KPI / Forecast / Risk run sequentially. Segmentation runs only when
    ``include_segmentation=True``. A failure in any step is
    captured in ``pipeline_errors`` and does not prevent subsequent steps
    from executing. The response always returns HTTP 200 when the request
    itself is valid; per-step failures are surfaced inside the payload.

    Raises HTTP 400 for an unrecognised ``business_type``.
    Raises HTTP 500 for unrecoverable aggregation or persistence failures.
    """
    assert_entity_allowed_for_tenant(entity_name=body.entity_name, security=security)
    now = datetime.now(tz=timezone.utc)
    period_start = now - timedelta(days=90)
    monthly_windows = _generate_monthly_windows(period_start, now)

    pipeline_errors: list[str] = []
    all_kpi_results: list[KPIRunResult] = []
    kpi_result: KPIRunResult | None = None
    kpi_summary: dict | None = None
    forecast_summary: dict | None = None
    risk_summary: dict | None = None
    seg_summary: dict | None = None

    # --- Step 1: KPI recomputation (per-month windows) ---
    try:
        orchestrator = KPIOrchestrator()
        for win_start, win_end in monthly_windows:
            try:
                result = orchestrator.run(
                    entity_name=body.entity_name,
                    business_type=body.business_type,
                    period_start=win_start,
                    period_end=win_end,
                    db=db,
                )
                all_kpi_results.append(result)
            except (KPIAggregationError, KPIPersistenceError):
                raise  # fatal — let outer handler deal with it
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "KPI recompute window failed entity=%r [%s, %s]: %s",
                    body.entity_name, win_start.isoformat(), win_end.isoformat(), exc,
                )
        if all_kpi_results:
            kpi_result = all_kpi_results[-1]  # latest month for downstream use
            kpi_summary = {
                "record_id": str(kpi_result.record_id),
                "has_errors": kpi_result.has_errors,
                "metrics": kpi_result.metrics,
                "windows_computed": len(all_kpi_results),
            }
        logger.info(
            "KPI recomputed entity=%r business_type=%r windows=%d succeeded=%d",
            body.entity_name, body.business_type,
            len(monthly_windows), len(all_kpi_results),
        )
    except KPIUnknownBusinessTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=build_error_detail(
                code=SCHEMA_CONFLICT,
                message=f"Unsupported business_type: {body.business_type}",
            ),
        ) from exc
    except (KPIAggregationError, KPIPersistenceError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(
                code=INTERNAL_FAILURE,
                message=f"KPI pipeline failed: {exc}",
            ),
        ) from exc

    # --- Step 2: Forecast (uses all monthly metric values) ---
    try:
        metric_name = primary_metric_for_business_type(body.business_type)
        values: list[float] = []
        for r in all_kpi_results:
            for v in _extract_primary_metric_values(r, metric_name):
                values.append(v)
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

    # --- Step 4: Segmentation (explicit only) ---
    if body.include_segmentation:
        try:
            from segmentation.orchestrator import SegmentationOrchestrator

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
