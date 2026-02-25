"""
app/services package marker with lazy exports.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CompetitorScrapingService",
    "get_competitor_scraping_service",
    "CSVHeaderValidationError",
    "CSVIngestionService",
    "CSVSchemaMappingError",
    "CSVPersistenceError",
    "get_csv_ingestion_service",
    "ExternalIngestionService",
    "get_external_ingestion_service",
    "IngestionTaskExecutor",
    "FastAPIBackgroundTaskExecutor",
    "IngestionOrchestratorService",
    "get_ingestion_orchestrator_service",
    "MacroIngestionError",
    "MacroIngestionValidationError",
    "MacroIngestionProviderError",
    "MacroIngestionService",
    "get_macro_ingestion_service",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CompetitorScrapingService": (
        "app.services.competitor_scraping_service",
        "CompetitorScrapingService",
    ),
    "get_competitor_scraping_service": (
        "app.services.competitor_scraping_service",
        "get_competitor_scraping_service",
    ),
    "CSVHeaderValidationError": (
        "app.services.csv_ingestion_service",
        "CSVHeaderValidationError",
    ),
    "CSVIngestionService": (
        "app.services.csv_ingestion_service",
        "CSVIngestionService",
    ),
    "CSVSchemaMappingError": (
        "app.services.csv_ingestion_service",
        "CSVSchemaMappingError",
    ),
    "CSVPersistenceError": (
        "app.services.csv_ingestion_service",
        "CSVPersistenceError",
    ),
    "get_csv_ingestion_service": (
        "app.services.csv_ingestion_service",
        "get_csv_ingestion_service",
    ),
    "ExternalIngestionService": (
        "app.services.external_ingestion_service",
        "ExternalIngestionService",
    ),
    "get_external_ingestion_service": (
        "app.services.external_ingestion_service",
        "get_external_ingestion_service",
    ),
    "IngestionTaskExecutor": (
        "app.services.ingestion_orchestrator_service",
        "IngestionTaskExecutor",
    ),
    "FastAPIBackgroundTaskExecutor": (
        "app.services.ingestion_orchestrator_service",
        "FastAPIBackgroundTaskExecutor",
    ),
    "IngestionOrchestratorService": (
        "app.services.ingestion_orchestrator_service",
        "IngestionOrchestratorService",
    ),
    "get_ingestion_orchestrator_service": (
        "app.services.ingestion_orchestrator_service",
        "get_ingestion_orchestrator_service",
    ),
    "MacroIngestionError": (
        "app.services.macro_ingestion_service",
        "MacroIngestionError",
    ),
    "MacroIngestionValidationError": (
        "app.services.macro_ingestion_service",
        "MacroIngestionValidationError",
    ),
    "MacroIngestionProviderError": (
        "app.services.macro_ingestion_service",
        "MacroIngestionProviderError",
    ),
    "MacroIngestionService": (
        "app.services.macro_ingestion_service",
        "MacroIngestionService",
    ),
    "get_macro_ingestion_service": (
        "app.services.macro_ingestion_service",
        "get_macro_ingestion_service",
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = __import__(module_name, fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

