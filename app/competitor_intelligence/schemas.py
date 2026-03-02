"""Structured data contracts for competitor intelligence flows."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class SearchRequest(BaseModel):
    """Input contract for a provider-backed web search."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=300)
    market: str = Field(default="US", min_length=2, max_length=32)
    language: str = Field(default="en", min_length=2, max_length=16)
    limit: int = Field(default=10, ge=1, le=50)
    recency_days: int | None = Field(default=None, ge=1, le=3650)
    use_cache: bool = True


class SourceDocument(BaseModel):
    """Normalized search result reference."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1, max_length=64)
    rank: int = Field(ge=1)
    url: HttpUrl
    title: str = Field(default="", max_length=1000)
    snippet: str = Field(default="", max_length=3000)
    published_at: datetime | None = None
    domain: str = Field(default="", max_length=255)


class SearchResponse(BaseModel):
    """Provider-agnostic search output."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1, max_length=64)
    query: str = Field(min_length=2, max_length=300)
    fetched_at: datetime
    cache_hit: bool = False
    documents: list[SourceDocument] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ScrapedDocument(BaseModel):
    """Raw scraper output normalized to text."""

    model_config = ConfigDict(extra="forbid")

    url: HttpUrl
    fetched_at: datetime
    status_code: int = Field(ge=0, le=999)
    title: str = Field(default="", max_length=1000)
    content_type: str = Field(default="", max_length=255)
    text: str = Field(default="", max_length=50000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ExtractionSignal(BaseModel):
    """One extracted competitor metric with provenance and confidence."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str = Field(min_length=1, max_length=120)
    value: float
    unit: str = Field(default="ratio", min_length=1, max_length=32)
    signal_type: Literal["competitor_metric", "industry_metric", "peer_metric"]
    confidence: float = Field(ge=0.0, le=1.0)
    source_url: HttpUrl
    evidence: str = Field(default="", max_length=500)
    timestamp: datetime | None = None


class ExtractionResult(BaseModel):
    """Strict structured extractor output."""

    model_config = ConfigDict(extra="forbid")

    competitor_name: str = Field(min_length=1, max_length=255)
    extraction_method: Literal["deterministic", "llm_structured"]
    extracted_at: datetime
    signals: list[ExtractionSignal] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CompetitorProfile(BaseModel):
    """Combined profile for one competitor from search+scrape+extract."""

    model_config = ConfigDict(extra="forbid")

    competitor_name: str = Field(min_length=1, max_length=255)
    queries: list[str] = Field(default_factory=list)
    search_documents: list[SourceDocument] = Field(default_factory=list)
    scraped_documents: list[ScrapedDocument] = Field(default_factory=list)
    extraction: ExtractionResult
    warnings: list[str] = Field(default_factory=list)


class AggregatedMarketMetric(BaseModel):
    """Deterministic market-level benchmark aggregate."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str = Field(min_length=1, max_length=120)
    unit: str = Field(default="ratio", min_length=1, max_length=32)
    sample_size: int = Field(ge=0)
    mean: float | None = None
    median: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    stdev: float | None = None


class CompetitorIntelligenceRequest(BaseModel):
    """Entry request for intelligence generation."""

    model_config = ConfigDict(extra="forbid")

    subject_entity: str = Field(min_length=1, max_length=255)
    competitors: list[str] = Field(min_length=1, max_length=50)
    market: str = Field(default="US", min_length=2, max_length=32)
    language: str = Field(default="en", min_length=2, max_length=16)
    recency_days: int | None = Field(default=180, ge=1, le=3650)
    documents_per_query: int = Field(default=8, ge=1, le=50)
    max_documents_per_competitor: int = Field(default=12, ge=1, le=50)
    max_scraped_urls_per_competitor: int = Field(default=6, ge=1, le=20)
    use_cache: bool = True


class CompetitorIntelligenceResponse(BaseModel):
    """Final response contract for the full module."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "partial", "failed"]
    generated_at: datetime
    subject_entity: str = Field(min_length=1, max_length=255)
    competitor_profiles: list[CompetitorProfile] = Field(default_factory=list)
    aggregated_market_data: list[AggregatedMarketMetric] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
