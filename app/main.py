from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from llm_synthesis.schema import FinalInsightResponse


def _validate_env() -> None:
    """
    Validate all required environment variables at startup.

    Runs before any service or database connection is initialised.
    Raises RuntimeError listing every missing or invalid variable so the
    operator can fix all problems in one restart cycle.

    Rules:
    - No empty-string values are accepted.
    - SQLite and local database fallbacks are not permitted.
    - LLM API key check is skipped only when LLM_ADAPTER=mock.
    - NEWS_API_KEY is required whenever NEWS_API_ENABLED is not false.
    """

    from db.config import load_env_files

    load_env_files()

    errors: list[str] = []

    # --- APP_MODE -------------------------------------------------------
    app_mode = os.getenv("APP_MODE", "").strip().lower()
    if not app_mode:
        errors.append(
            "APP_MODE is not set. It must be explicitly set to 'cloud'."
        )
    elif app_mode != "cloud":
        errors.append(
            f"APP_MODE='{app_mode}' is not valid. Allowed values: ['cloud']."
        )

    # --- Database URL ---------------------------------------------------
    database_url = os.getenv("DATABASE_URL", "").strip()
    cloud_database_url = os.getenv("CLOUD_DATABASE_URL", "").strip()
    if not database_url and not cloud_database_url:
        errors.append(
            "No database URL configured. Set DATABASE_URL or CLOUD_DATABASE_URL. "
            "SQLite and local database fallbacks are not permitted."
        )

    # --- LLM API key ----------------------------------------------------
    adapter = os.getenv("LLM_ADAPTER", "openai").strip().lower()
    if adapter != "mock":
        llm_api_key = os.getenv("LLM_API_KEY", "").strip()
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not llm_api_key and not openai_api_key:
            errors.append(
                "LLM API key is not set. Provide LLM_API_KEY or OPENAI_API_KEY. "
                "Empty strings are not permitted."
            )

    # --- News API key ---------------------------------------------------
    news_enabled_raw = os.getenv("NEWS_API_ENABLED", "true").strip().lower()
    news_enabled = news_enabled_raw in {"1", "true", "yes", "on"}
    if news_enabled and not os.getenv("NEWS_API_KEY", "").strip():
        errors.append(
            "NEWS_API_KEY is not set but NEWS_API_ENABLED is true. "
            "Set NEWS_API_KEY or disable the connector with NEWS_API_ENABLED=false."
        )

    if errors:
        raise RuntimeError(
            "Startup validation failed — missing or invalid environment variables:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


def _configure_logging() -> None:
    """
    Configure root logging once for the API process.
    """

    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _check_db() -> None:
    """Open a session and run SELECT 1. Raises RuntimeError if the DB is unreachable."""
    from sqlalchemy import text

    from db.session import SessionLocal

    try:
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
        finally:
            db.close()
    except Exception as exc:
        raise RuntimeError("Database unavailable.") from exc


def _check_schema() -> None:
    """
    Compare Base.metadata table names against the live DB schema.

    Every table registered on Base.metadata must exist in the database.
    If any are missing, log a critical error and abort startup so that
    the operator is forced to run migrations before serving traffic.

    Does NOT auto-migrate.
    """
    from sqlalchemy import inspect as sa_inspect

    import db.models  # noqa: F401 — registers all ORM models on Base.metadata
    from db.base import Base
    from db.session import get_engine

    inspector = sa_inspect(get_engine())
    actual: set[str] = set(inspector.get_table_names())
    expected: set[str] = set(Base.metadata.tables.keys())
    missing = expected - actual

    if missing:
        log = logging.getLogger(__name__)
        log.critical(
            "Schema mismatch — %d table(s) defined in ORM metadata are absent from "
            "the database: %s. Run 'alembic upgrade head' and restart.",
            len(missing),
            ", ".join(sorted(missing)),
        )
        raise RuntimeError(
            f"Schema mismatch: {len(missing)} table(s) missing from the database "
            f"({', '.join(sorted(missing))}). Run migrations and restart."
        )


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Validate DB connectivity and schema, start the scheduler on boot; shut it down on exit."""
    _check_db()
    logging.getLogger(__name__).info("Database connectivity confirmed")
    _check_schema()
    logging.getLogger(__name__).info("Database schema validated")

    from app.scheduler.jobs import build_scheduler

    scheduler = build_scheduler()
    scheduler.start()
    logging.getLogger(__name__).info("Scheduler started with %d jobs", len(scheduler.get_jobs()))
    try:
        yield
    finally:
        scheduler.shutdown(wait=True)
        logging.getLogger(__name__).info("Scheduler shut down")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """

    _validate_env()
    _configure_logging()

    application = FastAPI(
        title="InsightAgent API",
        version="1.0.0",
        lifespan=_lifespan,
    )

    from app.api.routers import (
        bi_export_router,
        client_router,
        competitor_scraping_router,
        csv_ingestion_router,
        external_ingestion_router,
        ingestion_orchestrator_router,
        kpi_router,
    )

    application.include_router(client_router)
    application.include_router(competitor_scraping_router)
    application.include_router(csv_ingestion_router)
    application.include_router(external_ingestion_router)
    application.include_router(ingestion_orchestrator_router)
    application.include_router(kpi_router)
    application.include_router(bi_export_router)

    @application.get("/health")
    def healthcheck() -> FinalInsightResponse:
        return FinalInsightResponse(
            insight="Service healthy",
            evidence="API and scheduler lifecycle initialized successfully.",
            impact="System is ready to process insight requests.",
            recommended_action="Continue normal operations.",
            priority="low",
            confidence_score=1.0,
        )

    return application


app = create_app()
