"""Competitor intelligence module with deterministic-first orchestration."""

from app.competitor_intelligence.config import CompetitorIntelligenceConfig
from app.competitor_intelligence.factory import build_competitor_intelligence_service
from app.competitor_intelligence.research_service import (
    CompetitorResearchResult,
    CompetitorResearchService,
)
from app.competitor_intelligence.service import CompetitorIntelligenceService

__all__ = [
    "CompetitorIntelligenceConfig",
    "CompetitorResearchResult",
    "CompetitorResearchService",
    "CompetitorIntelligenceService",
    "build_competitor_intelligence_service",
]
