"""
Model package exports.

Import all SQLAlchemy models here so metadata registration and Alembic
autogeneration work without extra imports.
"""

from db.models.analytical_metric import AnalyticalMetric
from db.models.canonical_insight_record import CanonicalInsightRecord
from db.models.client import Client
from db.models.dataset import Dataset
from db.models.ingestion_job import IngestionJob
from db.models.insight import Insight
from db.models.mapping_config import MappingConfig

__all__ = [
    "Client",
    "Dataset",
    "IngestionJob",
    "AnalyticalMetric",
    "Insight",
    "CanonicalInsightRecord",
    "MappingConfig",
]
