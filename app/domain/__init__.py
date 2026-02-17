"""
app/domain package marker.
"""

from app.domain.canonical_insight import CanonicalInsightInput, IngestionSummary, RowValidationError
from app.domain.competitor_scraping import CompetitorScrapeSummary
from app.domain.external_ingestion import SourceIngestionSummary

__all__ = [
    "CanonicalInsightInput",
    "CompetitorScrapeSummary",
    "IngestionSummary",
    "RowValidationError",
    "SourceIngestionSummary",
]
