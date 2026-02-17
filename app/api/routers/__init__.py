"""
app/api/routers package marker.
"""

from app.api.routers.competitor_scraping import router as competitor_scraping_router
from app.api.routers.csv_ingestion import router as csv_ingestion_router
from app.api.routers.external_ingestion import router as external_ingestion_router
from app.api.routers.ingestion_orchestrator import router as ingestion_orchestrator_router

__all__ = [
    "competitor_scraping_router",
    "csv_ingestion_router",
    "external_ingestion_router",
    "ingestion_orchestrator_router",
]
