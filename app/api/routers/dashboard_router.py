"""
app/api/routers/dashboard_router.py

Aggregation endpoint for the React Intelligence Dashboard.

GET /api/dashboard?entity_name=...&business_type=...

Returns a single JSON payload combining KPIs, risk, forecasts, and
insights into the shape expected by the React IntelligenceDashboard
component.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from agent.graph import insight_graph
from app.failure_codes import INTERNAL_FAILURE, SCHEMA_CONFLICT, build_error_detail
from app.services.category_registry import (
    get_processing_strategy,
    primary_metric_for_business_type,
)
from app.services.competitive_benchmark_service import build_competitive_benchmark_snapshot
from db.repositories.kpi_repository import KPIRepository
from db.session import get_db
from forecast.repository import ForecastRepository
from llm_synthesis.schema import InsightOutput
from risk.repository import RiskRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["dashboard"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_RISK_TO_HEALTH: dict[str, str] = {
    "low": "Strong",
    "moderate": "Moderate",
    "high": "At Risk",
    "critical": "Critical",
}

_REVENUE_METRIC_NAMES: frozenset[str] = frozenset({
    "mrr", "revenue", "total_revenue", "monthly_revenue",
})


def _build_kpi_metrics(
    computed_kpis: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Flatten computed_kpis JSONB into {name: {value, unit, label}}."""
    result: dict[str, dict[str, Any]] = {}
    for name, entry in computed_kpis.items():
        if isinstance(entry, dict) and "value" in entry:
            result[name] = {
                "value": entry["value"],
                "unit": entry.get("unit", ""),
                "label": name.replace("_", " ").title(),
            }
        elif isinstance(entry, (int, float)):
            result[name] = {
                "value": entry,
                "unit": "",
                "label": name.replace("_", " ").title(),
            }
    return result


def _extract_revenue_trend(
    kpi_rows: list[Any],
) -> list[dict[str, Any]]:
    """Build revenue time-series from ComputedKPI rows.

    Returns ``[{period: str, value: float}]``.
    """
    points: list[dict[str, Any]] = []
    for row in kpi_rows:
        kpis = row.computed_kpis or {}
        revenue_value: float | None = None
        for metric_name in _REVENUE_METRIC_NAMES:
            entry = kpis.get(metric_name)
            if entry is None:
                continue
            if isinstance(entry, dict) and "value" in entry:
                revenue_value = float(entry["value"])
            elif isinstance(entry, (int, float)):
                revenue_value = float(entry)
            if revenue_value is not None:
                break
        if revenue_value is not None:
            period = row.period_end.strftime("%Y-%m") if row.period_end else ""
            points.append({"period": period, "value": revenue_value})
    return points


def _build_market_share(
    kpi_rows: list[Any],
    forecast_data: dict[str, Any] | None,
    primary_metric: str,
) -> list[dict[str, Any]]:
    """Build market share / projection chart data.

    Returns ``[{period: str, value: float, projected: bool}]``.
    Simulation projections are clearly marked with ``projected=True``.
    """
    points: list[dict[str, Any]] = []

    # Actual data points from KPI history
    for row in kpi_rows:
        kpis = row.computed_kpis or {}
        entry = kpis.get(primary_metric) or kpis.get("growth_rate")
        if entry is None:
            continue
        value = entry.get("value") if isinstance(entry, dict) else entry
        if isinstance(value, (int, float)):
            period = row.period_end.strftime("%Y-%m") if row.period_end else ""
            points.append({
                "period": period,
                "value": round(float(value), 2),
                "projected": False,
            })

    # Projected points from forecast (marked projected=True)
    if forecast_data and isinstance(forecast_data, dict):
        forecast_dict = forecast_data.get("forecast", {})
        if isinstance(forecast_dict, dict):
            for key in sorted(forecast_dict.keys()):
                value = forecast_dict[key]
                if isinstance(value, (int, float)):
                    points.append({
                        "period": key.replace("_", " ").title(),
                        "value": round(float(value), 2),
                        "projected": True,
                    })

    return points


_BUSINESS_TYPE_LABELS: dict[str, str] = {
    "saas": "B2B SaaS",
    "ecommerce": "E-Commerce",
    "agency": "Agency",
    "marketing_analytics": "Marketing Analytics",
    "healthcare": "Healthcare",
    "retail": "Retail",
    "financial_markets": "Financial Markets",
    "operations": "Operations",
    "general_timeseries": "General Timeseries",
}


def _build_classification(
    business_type: str,
    confidence: float,
) -> dict[str, Any]:
    """Derive a single classification object from business type.

    Confidence is normalized to 0–100.
    """
    label = _BUSINESS_TYPE_LABELS.get(
        business_type, business_type.replace("_", " ").title(),
    )
    return {
        "label": label,
        "category": "business_type",
        "confidence": round(confidence * 100),
    }


def _transform_insight(raw: dict[str, Any]) -> dict[str, Any]:
    """Transform an LLM InsightOutput into the structured insight format.

    Returns ``{title, description, impact_score (0–100), dimension}``.
    """
    competitive = raw.get("competitive_analysis", {})
    strategic = raw.get("strategic_recommendations", {})

    title = str(competitive.get("summary", "")).strip()
    market_position = str(competitive.get("market_position", "")).strip()
    relative = str(competitive.get("relative_performance", "")).strip()
    immediate = strategic.get("immediate_actions")
    first_action = (
        str(immediate[0]).strip()
        if isinstance(immediate, list) and immediate
        else ""
    )
    description = " ".join(item for item in (market_position, relative, first_action) if item)

    confidence = competitive.get("confidence", 0.0)
    impact_score = round(confidence * 100, 1)

    return {
        "title": title,
        "description": description,
        "impact_score": impact_score,
        "dimension": "competitive",
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get("/dashboard", summary="Aggregated dashboard data for React frontend")
def get_dashboard(
    entity_name: str = Query(..., description="Entity/client identifier."),
    business_type: str = Query(
        default="saas",
        description="Business type/category.",
    ),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return aggregated dashboard data combining KPIs, risk, forecasts, and insights.

    This endpoint queries pre-computed data from the database rather than
    re-running the full pipeline, making it fast enough for dashboard refresh.
    If no insight has been generated yet, the insights array will be empty.
    """
    processing_strategy = get_processing_strategy(business_type) or business_type

    now = datetime.now(tz=timezone.utc)
    period_start = now - timedelta(days=365)

    pipeline_errors: list[str] = []

    # --- KPIs ---
    kpi_repo = KPIRepository(db)
    kpi_rows = kpi_repo.get_kpis_by_period(
        period_start=period_start,
        period_end=now,
        entity_name=entity_name,
    )

    kpi_metrics: dict[str, Any] = {}
    if kpi_rows:
        latest_kpis = kpi_rows[-1].computed_kpis or {}
        kpi_metrics = _build_kpi_metrics(latest_kpis)
    else:
        pipeline_errors.append("No KPI data found for this entity.")

    revenue_trend = _extract_revenue_trend(kpi_rows)

    # --- Risk ---
    risk_repo = RiskRepository()
    risk_record = risk_repo.get_latest_risk(session=db, entity_name=entity_name)

    if risk_record:
        risk_score = risk_record.risk_score
        risk_level = _classify_risk(risk_score)
        health_index = max(0, 100 - risk_score)
        health_label = _RISK_TO_HEALTH.get(risk_level, "Unknown")
    else:
        health_index = 50
        health_label = "No Data"
        pipeline_errors.append("No risk score found for this entity.")

    # --- Forecast ---
    try:
        primary_metric = primary_metric_for_business_type(processing_strategy)
    except Exception:
        primary_metric = "mrr"

    forecast_repo = ForecastRepository(db)
    forecast_record = forecast_repo.get_latest_forecast(
        entity_name=entity_name,
        metric_name=primary_metric,
    )
    forecast_data = forecast_record.forecast_data if forecast_record else None

    market_share = _build_market_share(kpi_rows, forecast_data, primary_metric)

    # --- Insight (from graph, if data exists) ---
    raw_insights: list[dict[str, Any]] = []
    if kpi_rows:
        try:
            state = insight_graph.invoke({
                "user_query": f"Generate executive dashboard insight for {entity_name}",
                "business_type": processing_strategy,
                "entity_name": entity_name,
            })
            response_raw = state.get("final_response")
            if isinstance(response_raw, str):
                payload = json.loads(response_raw)
                insight_model = InsightOutput.model_validate(payload)
                raw_insights.append(insight_model.model_dump())
        except Exception as exc:
            logger.warning("Dashboard insight generation failed: %s", exc)
            pipeline_errors.append(f"Insight generation: {exc}")

    # Transform insights to structured format (scores normalized 0–100)
    insights = [_transform_insight(r) for r in raw_insights]

    # --- Classification (single object, confidence 0–100) ---
    raw_confidence = (
        raw_insights[0].get("competitive_analysis", {}).get("confidence", 0.85)
        if raw_insights
        else 0.85
    )
    try:
        confidence_value = float(raw_confidence)
    except (TypeError, ValueError):
        confidence_value = 0.85
    confidence_value = max(0.0, min(1.0, confidence_value))
    classification = _build_classification(processing_strategy, confidence_value)
    competitive_benchmark: dict[str, Any] | None = None
    if kpi_rows:
        try:
            competitive_benchmark = build_competitive_benchmark_snapshot(
                db=db,
                entity_name=entity_name,
                business_type=processing_strategy,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dashboard competitive benchmark failed: %s", exc)
            pipeline_errors.append(f"Competitive benchmark: {exc}")

    return {
        "entity_name": entity_name,
        "business_type": processing_strategy,
        "health_index": health_index,
        "health_label": health_label,
        "kpi_metrics": kpi_metrics,
        "revenue_trend": revenue_trend,
        "market_share": market_share,
        "classification": classification,
        "insights": insights,
        "competitive_benchmark": competitive_benchmark,
        "pipeline_errors": pipeline_errors if pipeline_errors else None,
    }


def _classify_risk(score: int) -> str:
    """Map risk score to level label."""
    if score <= 30:
        return "low"
    if score <= 60:
        return "moderate"
    if score <= 80:
        return "high"
    return "critical"
