"""
Compatibility model module.

Allows importing core models from `db.models.models`.
"""

from db.models.canonical_insight_record import CanonicalInsightRecord
from db.models.client import Client
from db.models.computed_kpi import ComputedKPI
from db.models.dataset import Dataset
from db.models.ingestion_job import IngestionJob
from db.models.mapping_config import MappingConfig

__all__ = [
    "Client",
    "Dataset",
    "IngestionJob",
    "CanonicalInsightRecord",
    "MappingConfig",
    "ComputedKPI",
]
