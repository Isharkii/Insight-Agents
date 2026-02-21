"""
app/api/routers package marker.
"""

from app.api.routers.client_router import router as client_router
from app.api.routers.competitor_scraping import router as competitor_scraping_router
from app.api.routers.csv_ingestion import router as csv_ingestion_router
from app.api.routers.external_ingestion import router as external_ingestion_router
from app.api.routers.ingestion_orchestrator import router as ingestion_orchestrator_router
from app.api.routers.kpi_router import router as kpi_router
from app.api.routers.bi_export_router import router as bi_export_router

__all__ = [
    "bi_export_router",
    "client_router",
    "competitor_scraping_router",
    "csv_ingestion_router",
    "external_ingestion_router",
    "ingestion_orchestrator_router",
    "kpi_router",
]
