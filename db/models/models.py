"""
Compatibility model module.

Allows importing core models from `db.models.models`.
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
    "ScoringRun",
    "ScoringSubject",
    "RelativeScore",
    "CompositeScore",
    "RankingResult",
    "ScoreSignalReference",
]
