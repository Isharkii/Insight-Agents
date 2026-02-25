"""
Model package exports.

Import all SQLAlchemy models here so metadata registration and Alembic
autogeneration work without extra imports.
"""

from db.models.canonical_insight_record import CanonicalInsightRecord
from db.models.client import Client
from db.models.computed_kpi import ComputedKPI
from db.models.dataset import Dataset
from db.models.benchmarks import (
    Benchmark,
    BenchmarkMetric,
    BenchmarkSnapshot,
    IndustryCategory,
)
from db.models.ingestion_job import IngestionJob
from db.models.macro_metrics import MacroMetric, MacroMetricRun
from db.models.mapping_config import MappingConfig
from db.models.scoring import (
    CompositeScore,
    RankingResult,
    RelativeScore,
    ScoreSignalReference,
    ScoringRun,
    ScoringSubject,
)
from forecast.repository import ForecastMetric
from risk.repository import BusinessRiskScore

__all__ = [
    "Client",
    "Dataset",
    "IndustryCategory",
    "Benchmark",
    "BenchmarkMetric",
    "BenchmarkSnapshot",
    "IngestionJob",
    "CanonicalInsightRecord",
    "MappingConfig",
    "ComputedKPI",
    "MacroMetricRun",
    "MacroMetric",
    "ForecastMetric",
    "BusinessRiskScore",
    "ScoringRun",
    "ScoringSubject",
    "RelativeScore",
    "CompositeScore",
    "RankingResult",
    "ScoreSignalReference",
]
