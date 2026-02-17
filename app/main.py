from __future__ import annotations

import logging
import os

from fastapi import FastAPI

from app.api.routers import (
    competitor_scraping_router,
    csv_ingestion_router,
    external_ingestion_router,
    ingestion_orchestrator_router,
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


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """

    _configure_logging()

    application = FastAPI(title="InsightAgent API", version="1.0.0")
    application.include_router(competitor_scraping_router)
    application.include_router(csv_ingestion_router)
    application.include_router(external_ingestion_router)
    application.include_router(ingestion_orchestrator_router)

    @application.get("/health")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()
