"""
Compatibility model module.

Allows importing core models from `db.models.models`.
"""

from db.models.analytical_metric import AnalyticalMetric
from db.models.canonical_insight_record import CanonicalInsightRecord
from db.models.client import Client
from db.models.dataset import Dataset
from db.models.insight import Insight

__all__ = ["Client", "Dataset", "AnalyticalMetric", "Insight", "CanonicalInsightRecord"]
