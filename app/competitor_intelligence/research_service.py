"""Competitor research orchestration service."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.competitor_intelligence.cache import AsyncTTLCache
from app.competitor_intelligence.extraction.competitor_intelligence_extractor import (
    CompetitorIntelligence,
)
from app.competitor_intelligence.search.interfaces import SearchProvider

logger = logging.getLogger(__name__)


class _ExtractorLike(Protocol):
    async def extract(self, *, raw_text: str, company_name: str) -> CompetitorIntelligence:
        """Extract normalized competitor intelligence from text."""


class _AsyncScraperLike(Protocol):
    async def fetch(self, url: str) -> Any:
        """Fetch one URL asynchronously."""


class _AsyncTextScraperLike(Protocol):
    async def fetch_text(self, url: str) -> str | None:
        """Fetch one URL and return clean text."""


class CompetitorResearchResult(BaseModel):
    """Unified service response."""

    model_config = ConfigDict(extra="forbid")

    company: str = Field(min_length=1, max_length=255)
    research_timestamp: str
    sources: list[str] = Field(default_factory=list)
    intelligence: CompetitorIntelligence
    confidence: float = Field(ge=0.0, le=1.0)


class CompetitorResearchService:
    """Orchestrates search -> scrape -> extract into one JSON payload."""

    def __init__(
        self,
        *,
        providers: Mapping[str, SearchProvider],
        scraper: _AsyncScraperLike | _AsyncTextScraperLike | Any,
        extractor: _ExtractorLike,
        provider_name: str = "brave",
        cache: AsyncTTLCache[object] | None = None,
        max_pages: int = 3,
        search_limit: int = 10,
        max_text_length: int = 12000,
    ) -> None:
        if max_pages <= 0:
            raise ValueError("max_pages must be > 0.")
        if search_limit <= 0:
            raise ValueError("search_limit must be > 0.")
        if max_text_length <= 0:
            raise ValueError("max_text_length must be > 0.")
        self._providers = dict(providers)
        self._scraper = scraper
        self._extractor = extractor
        self._provider_name = provider_name.strip().lower()
        self._cache = cache
        self._max_pages = int(max_pages)
        self._search_limit = int(search_limit)
        self._max_text_length = int(max_text_length)

    async def research(self, *, company: str, query: str | None = None) -> dict[str, Any]:
        """Run competitor research and return unified JSON payload."""
        started = time.perf_counter()
        company_name = str(company or "").strip()
        if not company_name:
            return self._fallback_response(company="unknown")

        provider = self._providers.get(self._provider_name)
        if provider is None:
            logger.error(
                "event=competitor_research provider_missing provider=%s company=%s",
                self._provider_name,
                company_name,
            )
            return self._fallback_response(company=company_name)

        search_query = (query or f"{company_name} SaaS pricing target segment key features").strip()
        cache_key = (
            f"competitor_research:{self._provider_name}:{company_name}:{search_query}:"
            f"{self._max_pages}:{self._search_limit}:{self._max_text_length}"
        )
        if self._cache is not None:
            cached = await self._cache.get(cache_key)
            if isinstance(cached, dict):
                logger.info(
                    "event=competitor_research cache_hit=true company=%s provider=%s",
                    company_name,
                    self._provider_name,
                )
                return cached

        logger.info(
            "event=competitor_research_started company=%s provider=%s query=%s",
            company_name,
            self._provider_name,
            search_query,
        )

        try:
            search_response = await asyncio.to_thread(
                provider.search,
                search_query,
                limit=self._search_limit,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "event=competitor_research_search_failed company=%s provider=%s error=%s",
                company_name,
                self._provider_name,
                exc,
            )
            return self._fallback_response(company=company_name)

        urls = self._dedupe_urls(
            item.get("url", "")
            for item in search_response.get("results", [])
            if isinstance(item, dict)
        )[: self._max_pages]
        if not urls:
            logger.warning(
                "event=competitor_research_no_sources company=%s provider=%s",
                company_name,
                self._provider_name,
            )
            return self._fallback_response(company=company_name)

        scraped_texts = await self._fetch_sources(urls)
        joined_text = self._prepare_text(scraped_texts)
        if not joined_text:
            logger.warning(
                "event=competitor_research_no_text company=%s provider=%s sources=%d",
                company_name,
                self._provider_name,
                len(urls),
            )
            return self._fallback_response(company=company_name, sources=urls)

        try:
            intelligence = await self._extractor.extract(
                raw_text=joined_text,
                company_name=company_name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "event=competitor_research_extract_failed company=%s provider=%s error=%s",
                company_name,
                self._provider_name,
                exc,
            )
            return self._fallback_response(company=company_name, sources=urls)

        result = CompetitorResearchResult(
            company=company_name,
            research_timestamp=datetime.now(timezone.utc).isoformat(),
            sources=urls,
            intelligence=intelligence,
            confidence=float(intelligence.confidence_score),
        ).model_dump()

        if self._cache is not None:
            await self._cache.set(cache_key, result)

        logger.info(
            "event=competitor_research_finished company=%s provider=%s sources=%d elapsed_ms=%.2f confidence=%.3f",
            company_name,
            self._provider_name,
            len(urls),
            (time.perf_counter() - started) * 1000.0,
            result["confidence"],
        )
        return result

    async def _fetch_sources(self, urls: list[str]) -> list[str]:
        async def _fetch_one(url: str) -> str | None:
            # Supports both scraper.fetch_text(url) and scraper.fetch(url) contracts.
            if hasattr(self._scraper, "fetch_text"):
                return await self._scraper.fetch_text(url)
            if hasattr(self._scraper, "fetch"):
                fetched = await self._scraper.fetch(url)
                if isinstance(fetched, str):
                    return fetched
                if isinstance(fetched, dict):
                    text_candidate = fetched.get("text")
                    return str(text_candidate).strip() if text_candidate else None
                text_attr = getattr(fetched, "text", None)
                return str(text_attr).strip() if text_attr else None
            return None

        results = await asyncio.gather(*(_fetch_one(url) for url in urls), return_exceptions=True)
        texts: list[str] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("event=competitor_research_scrape_error error=%s", result)
                continue
            if result and result.strip():
                texts.append(result.strip())
        return texts

    def _prepare_text(self, texts: list[str]) -> str:
        if not texts:
            return ""
        merged = "\n\n".join(texts)
        return merged[: self._max_text_length]

    def _fallback_response(self, *, company: str, sources: list[str] | None = None) -> dict[str, Any]:
        fallback_intelligence = CompetitorIntelligence(
            company_name=company,
            pricing_model=None,
            target_segment=None,
            key_features=[],
            positioning=None,
            funding_status=None,
            estimated_scale=None,
            confidence_score=0.0,
        )
        return CompetitorResearchResult(
            company=company,
            research_timestamp=datetime.now(timezone.utc).isoformat(),
            sources=list(sources or []),
            intelligence=fallback_intelligence,
            confidence=0.0,
        ).model_dump()

    @staticmethod
    def _dedupe_urls(urls: Any) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for raw in urls:
            url = str(raw or "").strip()
            if not url:
                continue
            if url in seen:
                continue
            seen.add(url)
            output.append(url)
        return output
