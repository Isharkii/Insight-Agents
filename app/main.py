from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from llm_synthesis.schema import FinalInsightResponse

def _configure_logging() -> None:
    """
    Configure root logging once for the API process.
    """

    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Start the background scheduler on boot; shut it down gracefully on exit."""
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

    _configure_logging()

    application = FastAPI(
        title="InsightAgent API",
        version="1.0.0",
        lifespan=_lifespan,
    )

    from app.api.routers import (
        bi_export_router,
        competitor_scraping_router,
        csv_ingestion_router,
        external_ingestion_router,
        ingestion_orchestrator_router,
        kpi_router,
    )

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
