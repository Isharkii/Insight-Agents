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
  weekly_segmentation — 03:30 every Monday

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

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.orm import Session

from app.services.csv_ingestion_service import (
    _PRIMARY_METRIC_BY_BUSINESS_TYPE,
    _build_segmentation_records,
    _extract_primary_metric_values,
)
from app.services.kpi_orchestrator import KPIOrchestrator, KPIRunResult
from db.models.client import Client
from db.session import SessionLocal
from forecast.orchestrator import ForecastOrchestrator
from risk.orchestrator import RiskOrchestrator
from segmentation.orchestrator import SegmentationOrchestrator

logger = logging.getLogger(__name__)

_VALID_BUSINESS_TYPES: frozenset[str] = frozenset({"saas", "ecommerce", "agency"})


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
        if btype not in _VALID_BUSINESS_TYPES:
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
        rows = (
            session.query(Client.name, Client.config)
            .filter(
                Client.is_active.is_(True),
                Client.config.isnot(None),
                Client.config["business_type"].astext.in_(list(_VALID_BUSINESS_TYPES)),
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


def run_daily_kpi() -> None:
    """
    Recompute KPIs for all known entities over the trailing 90-day window.
    KPIOrchestrator commits internally; no explicit commit is needed here.
    """
    logger.info("Scheduler: daily_kpi starting")
    now = datetime.now(tz=timezone.utc)
    period_start = now - timedelta(days=90)

    with _session_scope() as db:
        entities = _resolve_entities(db)
        if not entities:
            logger.warning("Scheduler: daily_kpi — no entities found, skipping")
            return

        for entity_name, business_type in entities:
            try:
                result: KPIRunResult = KPIOrchestrator().run(
                    entity_name=entity_name,
                    business_type=business_type,
                    period_start=period_start,
                    period_end=now,
                    db=db,
                )
                logger.info(
                    "Scheduler: daily_kpi entity=%r business_type=%r "
                    "record_id=%s has_errors=%s",
                    entity_name,
                    business_type,
                    result.record_id,
                    result.has_errors,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Scheduler: daily_kpi failed entity=%r: %s", entity_name, exc
                )

    logger.info("Scheduler: daily_kpi complete")


# ---------------------------------------------------------------------------
# Job: Daily forecast update
# ---------------------------------------------------------------------------


def run_daily_forecast() -> None:
    """
    Regenerate forecasts for all known entities using the primary KPI metric.
    Commits per entity on success; rolls back on failure.
    """
    logger.info("Scheduler: daily_forecast starting")
    now = datetime.now(tz=timezone.utc)
    period_start = now - timedelta(days=90)

    with _session_scope() as db:
        entities = _resolve_entities(db)
        if not entities:
            logger.warning("Scheduler: daily_forecast — no entities found, skipping")
            return

        for entity_name, business_type in entities:
            try:
                # Re-run KPI to get a fresh metric value for the regression seed.
                kpi_result: KPIRunResult = KPIOrchestrator().run(
                    entity_name=entity_name,
                    business_type=business_type,
                    period_start=period_start,
                    period_end=now,
                    db=db,
                )
                metric_name = _PRIMARY_METRIC_BY_BUSINESS_TYPE.get(business_type, "revenue")
                values = _extract_primary_metric_values(kpi_result, metric_name)
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
                        "Scheduler: daily_forecast entity=%r trend=%s",
                        entity_name,
                        result.get("trend"),
                    )
            except Exception as exc:  # noqa: BLE001
                db.rollback()
                logger.warning(
                    "Scheduler: daily_forecast failed entity=%r: %s", entity_name, exc
                )

    logger.info("Scheduler: daily_forecast complete")


# ---------------------------------------------------------------------------
# Job: Daily risk update
# ---------------------------------------------------------------------------


def run_daily_risk() -> None:
    """
    Recompute risk scores for all known entities.
    Commits per entity on success; rolls back on failure.
    """
    logger.info("Scheduler: daily_risk starting")

    with _session_scope() as db:
        entities = _resolve_entities(db)
        if not entities:
            logger.warning("Scheduler: daily_risk — no entities found, skipping")
            return

        for entity_name, _ in entities:
            try:
                result = RiskOrchestrator(db).generate_risk_score(
                    entity_name=entity_name,
                    kpi_data={},
                    forecast_data={},
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
                logger.warning(
                    "Scheduler: daily_risk failed entity=%r: %s", entity_name, exc
                )

    logger.info("Scheduler: daily_risk complete")


# ---------------------------------------------------------------------------
# Job: Weekly segmentation refresh
# ---------------------------------------------------------------------------


def run_weekly_segmentation() -> None:
    """
    Re-cluster all known entities using their most recent KPI metrics.
    Commits per entity on success; rolls back on failure.
    """
    logger.info("Scheduler: weekly_segmentation starting")
    now = datetime.now(tz=timezone.utc)
    period_start = now - timedelta(days=90)

    with _session_scope() as db:
        entities = _resolve_entities(db)
        if not entities:
            logger.warning("Scheduler: weekly_segmentation — no entities found, skipping")
            return

        for entity_name, business_type in entities:
            try:
                kpi_result: KPIRunResult = KPIOrchestrator().run(
                    entity_name=entity_name,
                    business_type=business_type,
                    period_start=period_start,
                    period_end=now,
                    db=db,
                )
                seg_records = _build_segmentation_records(kpi_result)
                n_clusters = min(3, len(seg_records))
                if n_clusters < 1:
                    logger.info(
                        "Scheduler: weekly_segmentation skipped entity=%r: "
                        "insufficient records",
                        entity_name,
                    )
                    continue
                result = SegmentationOrchestrator(session=db).run_segmentation(
                    entity_name=entity_name,
                    records=seg_records,
                    n_clusters=n_clusters,
                )
                db.commit()
                logger.info(
                    "Scheduler: weekly_segmentation entity=%r n_clusters=%s",
                    entity_name,
                    result.get("n_clusters"),
                )
            except Exception as exc:  # noqa: BLE001
                db.rollback()
                logger.warning(
                    "Scheduler: weekly_segmentation failed entity=%r: %s",
                    entity_name,
                    exc,
                )

    logger.info("Scheduler: weekly_segmentation complete")


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
        weekly_segmentation — 03:30 every Monday
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
    scheduler.add_job(
        run_weekly_segmentation,
        trigger="cron",
        day_of_week="mon",
        hour=3,
        minute=30,
        id="weekly_segmentation",
        name="Weekly segmentation refresh",
        replace_existing=True,
        misfire_grace_time=7200,
    )

    return scheduler
