"""Factory wiring for competitor intelligence module dependencies."""

from __future__ import annotations

import os

from app.competitor_intelligence.cache import AsyncTTLCache
from app.competitor_intelligence.config import CompetitorIntelligenceConfig
from app.competitor_intelligence.extraction import (
    AsyncOpenAIJsonClient,
    DeterministicExtractor,
    LLMStructuredExtractor,
)
from app.competitor_intelligence.interfaces import Extractor, SearchProvider
from app.competitor_intelligence.providers import (
    BraveSearchProvider,
    SerperSearchProvider,
    TavilySearchProvider,
)
from app.competitor_intelligence.scraping import RequestsScraper
from app.competitor_intelligence.service import CompetitorIntelligenceService


def build_competitor_intelligence_service(
    config: CompetitorIntelligenceConfig | None = None,
) -> CompetitorIntelligenceService:
    """Build fully wired service with swappable provider/extractor layers."""
    cfg = config or CompetitorIntelligenceConfig.from_env()
    search_provider = _build_search_provider(cfg)
    scraper = RequestsScraper(config=cfg.scraper)
    extractor = _build_extractor(cfg)
    cache = None
    if cfg.cache.enabled:
        cache = AsyncTTLCache[object](
            ttl_seconds=cfg.cache.ttl_seconds,
            max_size=cfg.cache.max_size,
        )
    return CompetitorIntelligenceService(
        config=cfg,
        search_provider=search_provider,
        scraper=scraper,
        extractor=extractor,
        cache=cache,
    )


def _build_search_provider(config: CompetitorIntelligenceConfig) -> SearchProvider:
    provider_name = config.provider
    if provider_name == "brave":
        return BraveSearchProvider(config=config.brave)
    if provider_name == "serper":
        return SerperSearchProvider(config=config.serper)
    if provider_name == "tavily":
        return TavilySearchProvider(config=config.tavily)
    raise ValueError(f"Unsupported search provider: {provider_name}")


def _build_extractor(config: CompetitorIntelligenceConfig) -> Extractor:
    if config.extraction.mode == "deterministic":
        return DeterministicExtractor()

    api_key = os.getenv("COMP_INTEL_LLM_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("COMP_INTEL_LLM_API_KEY or OPENAI_API_KEY is required for llm_structured extractor mode.")
    model = (
        str(os.getenv("COMP_INTEL_LLM_MODEL", "gpt-5.4") or "").strip()
        or "gpt-5.4"
    )
    client = AsyncOpenAIJsonClient(model=model, api_key=api_key)
    return LLMStructuredExtractor(
        llm_client=client,
        max_docs=config.extraction.max_docs,
        max_chars_per_doc=config.extraction.max_chars_per_doc,
    )
