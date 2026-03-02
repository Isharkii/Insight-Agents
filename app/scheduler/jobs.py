"""
app/scheduler/jobs.py

APScheduler-based batch scheduler for periodic analytics recomputation.

Entity discovery (no hardcoded names)
--------------------------------------
Entities are resolved at job runtime from two sources, merged and deduplicated:

  1. ``Client`` table — active clients whose ``config`` JSONB contains a
     ``"business_type"`` key (value must be "saas", "ecommerce", or "agency").
  2. ``SCHEDULER_ENTITIES`` env var — comma-separated ``name:type`` pairs,
     e.g. ``acme:saas,globex:ecommerce``. Acts as an explicit override / seed
     for entities not yet registered in the clients table.

Schedule (all times UTC)
--------------------------
  daily_kpi          — 02:00 every day
  daily_forecast     — 02:30 every day
  daily_risk         — 03:00 every day

Lifecycle
----------
Call ``build_scheduler()`` once to get a configured ``BackgroundScheduler``.
Start it on app boot; shut it down gracefully on app shutdown.
The scheduler is wired into FastAPI via the ``lifespan`` context in main.py.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Iterator

from dateutil.relativedelta import relativedelta

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import select
from sqlalchemy.orm import Session

from llm_synthesis.schema import FinalInsightResponse
from app.services.category_registry import (
    churn_metric_for_business_type,
    primary_metric_for_business_type,
    supported_categories,
)
from app.services.kpi_orchestrator import KPIOrchestrator, KPIRunResult
from db.models.client import Client
from db.models.computed_kpi import ComputedKPI
from db.session import SessionLocal
from forecast.orchestrator import ForecastOrchestrator
from forecast.repository import ForecastRepository
from risk.orchestrator import RiskOrchestrator

logger = logging.getLogger(__name__)

def _valid_business_types() -> frozenset[str]:
    try:
        values = tuple(supported_categories())
        if values:
            return frozenset(values)
    except Exception:  # noqa: BLE001
        logger.exception("Unable to load supported categories from registry.")
    return frozenset({"saas", "ecommerce", "agency"})


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
    while start < period_end:
        end = start + relativedelta(months=1)
        if end > period_end:
            end = period_end
        windows.append((start, end))
        start = start + relativedelta(months=1)
    return windows


def _job_success(stage_name: str) -> FinalInsightResponse:
    return FinalInsightResponse(
        insight=f"{stage_name} completed",
        evidence=f"Scheduler job '{stage_name}' finished successfully.",
        impact="Scheduled analytics state remains current.",
        recommended_action="No action required.",
        priority="low",
        confidence_score=1.0,
    )


def _job_failure(stage_name: str, error: Exception) -> FinalInsightResponse:
    logger.exception("Scheduler job failed stage=%s", stage_name)
    return FinalInsightResponse.failure(
        reason=f"Scheduler stage '{stage_name}' failed: {type(error).__name__}: {error}"
    )


# ---------------------------------------------------------------------------
# Entity discovery
# ---------------------------------------------------------------------------


def _entities_from_env() -> list[tuple[str, str]]:
    """
    Parse ``SCHEDULER_ENTITIES`` env var into (entity_name, business_type) pairs.

    Format: ``name:type,name:type``  (whitespace-tolerant, case-insensitive type).
    Invalid or unknown business types are skipped with a WARNING log.
    """
    raw = os.getenv("SCHEDULER_ENTITIES", "").strip()
    if not raw:
        return []
    valid_business_types = _valid_business_types()
    entities: list[tuple[str, str]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":", 1)
        if len(parts) != 2:
            logger.warning("SCHEDULER_ENTITIES: skipping malformed token %r", token)
            continue
        name, btype = parts[0].strip(), parts[1].strip().lower()
        if not name:
            logger.warning("SCHEDULER_ENTITIES: skipping token with empty name %r", token)
            continue
        if btype not in valid_business_types:
            logger.warning(
                "SCHEDULER_ENTITIES: skipping %r — unknown business_type %r", name, btype
            )
            continue
        entities.append((name, btype))
    return entities


def _entities_from_db(session: Session) -> list[tuple[str, str]]:
    """
    Query active ``Client`` rows whose ``config`` JSONB has a valid
    ``"business_type"`` key.  Returns (client.name, business_type) pairs.
    """
    try:
        valid_business_types = list(_valid_business_types())
        rows = (
            session.query(Client.name, Client.config)
            .filter(
                Client.is_active.is_(True),
                Client.config.isnot(None),
                Client.config["business_type"].astext.in_(valid_business_types),
            )
            .all()
        )
        return [(row.name, row.config["business_type"]) for row in rows]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Entity discovery from DB failed: %s", exc)
        return []


def _resolve_entities(session: Session) -> list[tuple[str, str]]:
    """
    Merge env-var entities and DB entities; env-var entries take precedence
    for duplicate names.  Returns deduplicated (entity_name, business_type) list.
    """
    env_entities = _entities_from_env()
    db_entities = _entities_from_db(session)

    # Env-var wins on name collision.
    merged: dict[str, str] = {name: btype for name, btype in db_entities}
    for name, btype in env_entities:
        merged[name] = btype

    return list(merged.items())


# ---------------------------------------------------------------------------
# Session helper
# ---------------------------------------------------------------------------


@contextmanager
def _session_scope() -> Iterator[Session]:
    """Yield a fresh session and ensure it is closed on exit."""
    session: Session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Job: Daily KPI recomputation
# ---------------------------------------------------------------------------


def run_daily_kpi() -> FinalInsightResponse:
    """
    Recompute KPIs for all known entities using per-month windows so that
    downstream nodes receive a proper time-series instead of a single aggregate.
    KPIOrchestrator commits internally; no explicit commit is needed here.
    """
    try:
        logger.info("Scheduler: daily_kpi starting")
        now = datetime.now(tz=timezone.utc)
        period_start = now - timedelta(days=90)
        monthly_windows = _generate_monthly_windows(period_start, now)

        with _session_scope() as db:
            entities = _resolve_entities(db)
            if not entities:
                logger.warning("Scheduler: daily_kpi — no entities found, skipping")
                return _job_success("daily_kpi")

            orchestrator = KPIOrchestrator()
            for entity_name, business_type in entities:
                succeeded = 0
                for win_start, win_end in monthly_windows:
                    try:
                        orchestrator.run(
                            entity_name=entity_name,
                            business_type=business_type,
                            period_start=win_start,
                            period_end=win_end,
                            db=db,
                        )
                        succeeded += 1
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Scheduler: daily_kpi failed entity=%r window=[%s, %s]: %s",
                            entity_name, win_start.isoformat(), win_end.isoformat(), exc,
                        )
                logger.info(
                    "Scheduler: daily_kpi entity=%r business_type=%r windows=%d succeeded=%d",
                    entity_name, business_type, len(monthly_windows), succeeded,
                )

        logger.info("Scheduler: daily_kpi complete")
        return _job_success("daily_kpi")
    except Exception as exc:  # noqa: BLE001
        return _job_failure("daily_kpi", exc)


# ---------------------------------------------------------------------------
# Job: Daily forecast update
# ---------------------------------------------------------------------------


def run_daily_forecast() -> FinalInsightResponse:
    """
    Regenerate forecasts for all known entities using the primary KPI metric.
    Collects per-month KPI values to feed the forecast with a proper time-series.
    Commits per entity on success; rolls back on failure.
    """
    try:
        logger.info("Scheduler: daily_forecast starting")
        now = datetime.now(tz=timezone.utc)
        period_start = now - timedelta(days=90)
        monthly_windows = _generate_monthly_windows(period_start, now)

        with _session_scope() as db:
            entities = _resolve_entities(db)
            if not entities:
                logger.warning("Scheduler: daily_forecast — no entities found, skipping")
                return _job_success("daily_forecast")

            orchestrator = KPIOrchestrator()
            for entity_name, business_type in entities:
                try:
                    metric_name = primary_metric_for_business_type(business_type)
                    values: list[float] = []
                    for win_start, win_end in monthly_windows:
                        try:
                            kpi_result = orchestrator.run(
                                entity_name=entity_name,
                                business_type=business_type,
                                period_start=win_start,
                                period_end=win_end,
                                db=db,
                            )
                            for v in _extract_primary_metric_values(kpi_result, metric_name):
                                values.append(v)
                        except Exception:  # noqa: BLE001
                            pass  # window-level failure is non-fatal

                    result = ForecastOrchestrator(db).generate_forecast(
                        entity_name=entity_name,
                        metric_name=metric_name,
                        values=values,
                    )
                    if "error" in result:
                        logger.info(
                            "Scheduler: daily_forecast deferred entity=%r: %s",
                            entity_name,
                            result["error"],
                        )
                    else:
                        db.commit()
                        logger.info(
                            "Scheduler: daily_forecast entity=%r trend=%s points=%d",
                            entity_name,
                            result.get("trend"),
                            len(values),
                        )
                except Exception as exc:  # noqa: BLE001
                    db.rollback()
                    logger.warning(
                        "Scheduler: daily_forecast failed entity=%r: %s", entity_name, exc
                    )

        logger.info("Scheduler: daily_forecast complete")
        return _job_success("daily_forecast")
    except Exception as exc:  # noqa: BLE001
        return _job_failure("daily_forecast", exc)


# ---------------------------------------------------------------------------
# Job: Daily risk update
# ---------------------------------------------------------------------------


def run_daily_risk() -> FinalInsightResponse:
    """
    Recompute risk scores for all known entities.
    Commits per entity on success; rolls back on failure.
    """
    try:
        logger.info("Scheduler: daily_risk starting")

        with _session_scope() as db:
            entities = _resolve_entities(db)
            if not entities:
                logger.warning("Scheduler: daily_risk — no entities found, skipping")
                return _job_success("daily_risk")

            for entity_name, business_type in entities:
                try:
                    # Fetch the most recent ComputedKPI row for this entity.
                    kpi_record = db.scalars(
                        select(ComputedKPI)
                        .where(ComputedKPI.entity_name == entity_name)
                        .order_by(ComputedKPI.period_end.desc())
                        .limit(1)
                    ).first()
                    if kpi_record is None:
                        raise ValueError(
                            f"No KPI data found for entity={entity_name!r}; cannot compute risk."
                        )

                    # Fetch the most recent forecast for the primary metric.
                    metric_name = primary_metric_for_business_type(business_type)
                    forecast_record = ForecastRepository(db).get_latest_forecast(
                        entity_name=entity_name,
                        metric_name=metric_name,
                    )
                    if forecast_record is None:
                        raise ValueError(
                            f"No forecast data found for entity={entity_name!r} "
                            f"metric={metric_name!r}; cannot compute risk."
                        )

                    # Map the stored computed_kpis JSONB to the flat kpi_data dict.
                    # JSONB shape: {"metric_key": {"value": float | None, "unit": str, ...}}
                    raw_kpis: dict = kpi_record.computed_kpis or {}

                    def _kpi_value(key: str) -> float:
                        entry = raw_kpis.get(key) or {}
                        v = entry.get("value")
                        return float(v) if v is not None else 0.0

                    churn_key = churn_metric_for_business_type(business_type)
                    kpi_data: dict = {
                        "revenue_growth_delta": _kpi_value("growth_rate"),
                        "churn_delta":          _kpi_value(churn_key),
                        "conversion_delta":     _kpi_value("conversion_rate"),
                    }

                    # forecast_data is the full JSONB payload stored by ForecastOrchestrator.
                    # It already contains slope, deviation_percentage, and churn_acceleration.
                    forecast_data: dict = forecast_record.forecast_data or {}

                    result = RiskOrchestrator(db).generate_risk_score(
                        entity_name=entity_name,
                        kpi_data=kpi_data,
                        forecast_data=forecast_data,
                    )
                    db.commit()
                    logger.info(
                        "Scheduler: daily_risk entity=%r score=%s level=%s",
                        entity_name,
                        result.get("risk_score"),
                        result.get("risk_level"),
                    )
                except Exception as exc:  # noqa: BLE001
                    db.rollback()
                    logger.error(
                        "Scheduler: daily_risk critical failure entity=%r: %s",
                        entity_name,
                        exc,
                        extra={
                            "event": "scheduler_daily_risk_critical_failure",
                            "entity_name": entity_name,
                        },
                        exc_info=True,
                    )
                    raise

        logger.info("Scheduler: daily_risk complete")
        return _job_success("daily_risk")
    except Exception as exc:  # noqa: BLE001
        return _job_failure("daily_risk", exc)


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------


def build_scheduler() -> BackgroundScheduler:
    """
    Build and register all periodic jobs.

    Returns a configured but *not yet started* ``BackgroundScheduler``.
    The caller must call ``.start()`` and ``.shutdown(wait=True)`` at the
    appropriate lifecycle points.

    Schedule (UTC):
        daily_kpi           — 02:00 every day
        daily_forecast      — 02:30 every day
        daily_risk          — 03:00 every day
    """
    scheduler = BackgroundScheduler(timezone="UTC")

    scheduler.add_job(
        run_daily_kpi,
        trigger="cron",
        hour=2,
        minute=0,
        id="daily_kpi",
        name="Daily KPI recomputation",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    scheduler.add_job(
        run_daily_forecast,
        trigger="cron",
        hour=2,
        minute=30,
        id="daily_forecast",
        name="Daily forecast update",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    scheduler.add_job(
        run_daily_risk,
        trigger="cron",
        hour=3,
        minute=0,
        id="daily_risk",
        name="Daily risk scoring",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    return scheduler
