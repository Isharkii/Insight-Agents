"""Deterministic-first competitor intelligence orchestration service."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, median, pstdev
from typing import Any, Iterable
from urllib.parse import urlparse

from app.competitor_intelligence.cache import AsyncTTLCache
from app.competitor_intelligence.config import CompetitorIntelligenceConfig
from app.competitor_intelligence.interfaces import Extractor, Scraper, SearchProvider
from app.competitor_intelligence.schemas import (
    AggregatedMarketMetric,
    CompetitorIntelligenceRequest,
    CompetitorIntelligenceResponse,
    CompetitorProfile,
    SearchRequest,
    SearchResponse,
)

logger = logging.getLogger(__name__)


class CompetitorIntelligenceService:
    """Main service composing provider, scraper, extractor, and cache."""

    def __init__(
        self,
        *,
        config: CompetitorIntelligenceConfig,
        search_provider: SearchProvider,
        scraper: Scraper,
        extractor: Extractor,
        cache: AsyncTTLCache[object] | None = None,
    ) -> None:
        self._config = config
        self._search_provider = search_provider
        self._scraper = scraper
        self._extractor = extractor
        self._cache = cache

    async def generate(self, request: CompetitorIntelligenceRequest) -> CompetitorIntelligenceResponse:
        started_at = time.perf_counter()
        warnings: list[str] = []
        try:
            competitors = _normalize_competitors(request.competitors)
            if not competitors:
                return CompetitorIntelligenceResponse(
                    status="failed",
                    generated_at=datetime.now(timezone.utc),
                    subject_entity=request.subject_entity,
                    competitor_profiles=[],
                    aggregated_market_data=[],
                    warnings=["No valid competitors provided."],
                )

            logger.info(
                "competitor_intel_started subject=%s competitors=%d provider=%s",
                request.subject_entity,
                len(competitors),
                self._search_provider.name,
            )

            results = await asyncio.gather(
                *(self._build_profile(request=request, competitor_name=name) for name in competitors),
                return_exceptions=True,
            )
            profiles: list[CompetitorProfile] = []
            for index, result in enumerate(results):
                if isinstance(result, Exception):
                    competitor = competitors[index] if index < len(competitors) else "unknown"
                    warnings.append(f"profile_build_failed:{competitor}:{result}")
                    continue
                profiles.append(result)

            aggregated = _aggregate_market_metrics(profiles)
            for profile in profiles:
                warnings.extend(profile.warnings)
                warnings.extend(profile.extraction.warnings)
            status = "success"
            if warnings:
                status = "partial"
            if not profiles:
                status = "failed"

            return CompetitorIntelligenceResponse(
                status=status,
                generated_at=datetime.now(timezone.utc),
                subject_entity=request.subject_entity,
                competitor_profiles=profiles,
                aggregated_market_data=aggregated,
                warnings=_dedupe(warnings),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("competitor_intel_failed subject=%s error=%s", request.subject_entity, exc)
            return CompetitorIntelligenceResponse(
                status="failed",
                generated_at=datetime.now(timezone.utc),
                subject_entity=request.subject_entity,
                competitor_profiles=[],
                aggregated_market_data=[],
                warnings=[f"unexpected_error:{exc}"],
            )
        finally:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
            cache_stats: dict[str, int] | None = None
            if self._cache is not None:
                cache_stats = await self._cache.stats()
            logger.info(
                "competitor_intel_finished subject=%s elapsed_ms=%s cache=%s",
                request.subject_entity,
                elapsed_ms,
                cache_stats or {},
            )

    async def _build_profile(
        self,
        *,
        request: CompetitorIntelligenceRequest,
        competitor_name: str,
    ) -> CompetitorProfile:
        competitor_started = time.perf_counter()
        queries = [template.format(competitor=competitor_name) for template in self._config.query_templates]
        search_started = time.perf_counter()
        search_results = await self._search_many(
            queries=queries,
            market=request.market,
            language=request.language,
            limit=request.documents_per_query,
            recency_days=request.recency_days,
            use_cache=request.use_cache,
        )
        search_elapsed_ms = round((time.perf_counter() - search_started) * 1000.0, 2)
        search_docs = _dedupe_documents(
            responses=search_results,
            max_docs=request.max_documents_per_competitor,
        )
        # Scraping defaults to top 3 results; configurable via config.max_pages.
        max_pages = max(1, min(self._config.max_pages, request.max_scraped_urls_per_competitor))
        candidate_urls = _dedupe_urls(str(doc.url) for doc in search_docs)
        urls = _filter_blacklisted_urls(candidate_urls, self._config.scraper.domain_blacklist)[:max_pages]
        blocked_count = max(0, len(candidate_urls) - len(urls))
        if blocked_count:
            logger.info(
                "competitor_intel_blacklist_applied competitor=%s blocked_urls=%d",
                competitor_name,
                blocked_count,
            )
        scrape_started = time.perf_counter()
        scraped_docs = await self._scrape_many(urls=urls, use_cache=request.use_cache)
        scrape_elapsed_ms = round((time.perf_counter() - scrape_started) * 1000.0, 2)
        extract_started = time.perf_counter()
        extraction = await self._extractor.extract(
            competitor_name=competitor_name,
            documents=scraped_docs,
        )
        extract_elapsed_ms = round((time.perf_counter() - extract_started) * 1000.0, 2)
        competitor_elapsed_ms = round((time.perf_counter() - competitor_started) * 1000.0, 2)
        local_warnings: list[str] = []
        for resp in search_results:
            local_warnings.extend(resp.warnings)

        logger.info(
            "competitor_intel_profile competitor=%s queries=%d unique_urls=%d scraped=%d signals=%d timings_ms={search:%s,scrape:%s,extract:%s,total:%s}",
            competitor_name,
            len(queries),
            len(urls),
            len(scraped_docs),
            len(extraction.signals),
            search_elapsed_ms,
            scrape_elapsed_ms,
            extract_elapsed_ms,
            competitor_elapsed_ms,
        )

        return CompetitorProfile(
            competitor_name=competitor_name,
            queries=queries,
            search_documents=search_docs,
            scraped_documents=scraped_docs,
            extraction=extraction,
            warnings=_dedupe(local_warnings),
        )

    async def _search_many(
        self,
        *,
        queries: Iterable[str],
        market: str,
        language: str,
        limit: int,
        recency_days: int | None,
        use_cache: bool,
    ) -> list[SearchResponse]:
        semaphore = asyncio.Semaphore(max(1, self._config.search_max_concurrency))

        async def _bounded_search(query: str) -> SearchResponse:
            async with semaphore:
                return await self._search_one(
                    SearchRequest(
                        query=query,
                        market=market,
                        language=language,
                        limit=limit,
                        recency_days=recency_days,
                        use_cache=use_cache,
                    )
                )

        tasks = [_bounded_search(query) for query in queries]
        return await asyncio.gather(*tasks)

    async def _search_one(self, request: SearchRequest) -> SearchResponse:
        cache_key = f"search:{self._search_provider.name}:{request.model_dump_json()}"
        if request.use_cache and self._cache is not None:
            cached = await self._cache.get(cache_key)
            if isinstance(cached, SearchResponse):
                return cached.model_copy(update={"cache_hit": True})
        response = await self._search_provider.search(request)
        if request.use_cache and self._cache is not None:
            await self._cache.set(cache_key, response)
        return response

    async def _scrape_many(self, *, urls: list[str], use_cache: bool) -> list[Any]:
        unique_urls = _dedupe_urls(urls)
        if not unique_urls:
            return []

        if not use_cache or self._cache is None:
            return await self._scraper.fetch_many(
                unique_urls,
                max_concurrency=self._config.scraper.max_concurrency,
            )

        cached_results = []
        missing_urls = []
        for url in unique_urls:
            cache_key = f"scrape:{url}"
            cached = await self._cache.get(cache_key)
            if cached is None:
                missing_urls.append(url)
                continue
            cached_results.append(cached)

        fetched = []
        if missing_urls:
            fetched = await self._scraper.fetch_many(
                missing_urls,
                max_concurrency=self._config.scraper.max_concurrency,
            )
            for doc in fetched:
                await self._cache.set(f"scrape:{doc.url}", doc)

        all_docs = [*cached_results, *fetched]
        docs_by_url = {str(doc.url): doc for doc in all_docs}
        return [docs_by_url[url] for url in unique_urls if url in docs_by_url]


def _normalize_competitors(names: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in names:
        name = str(raw).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(name)
    return cleaned


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _dedupe_urls(urls: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for raw in urls:
        url = str(raw).strip()
        if not url:
            continue
        if url in seen:
            continue
        seen.add(url)
        output.append(url)
    return output


def _filter_blacklisted_urls(urls: Iterable[str], blacklist: Iterable[str]) -> list[str]:
    normalized_blacklist = {item.strip().lower() for item in blacklist if item and item.strip()}
    if not normalized_blacklist:
        return list(urls)
    filtered: list[str] = []
    for url in urls:
        host = (urlparse(url).hostname or "").strip().lower()
        if not host:
            continue
        blocked = False
        for blocked_domain in normalized_blacklist:
            if host == blocked_domain or host.endswith(f".{blocked_domain}"):
                blocked = True
                break
        if not blocked:
            filtered.append(url)
    return filtered


def _dedupe_documents(*, responses: list[SearchResponse], max_docs: int) -> list:
    seen_urls: set[str] = set()
    docs = []
    for response in responses:
        for doc in response.documents:
            url = str(doc.url)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            docs.append(doc)
            if len(docs) >= max_docs:
                return docs
    return docs


def _aggregate_market_metrics(profiles: list[CompetitorProfile]) -> list[AggregatedMarketMetric]:
    values_by_metric: dict[tuple[str, str], list[float]] = defaultdict(list)
    for profile in profiles:
        for signal in profile.extraction.signals:
            key = (signal.metric_name, signal.unit)
            values_by_metric[key].append(float(signal.value))

    aggregates: list[AggregatedMarketMetric] = []
    for (metric_name, unit), values in sorted(values_by_metric.items()):
        if not values:
            continue
        aggregates.append(
            AggregatedMarketMetric(
                metric_name=metric_name,
                unit=unit,
                sample_size=len(values),
                mean=round(mean(values), 6),
                median=round(median(values), 6),
                min_value=round(min(values), 6),
                max_value=round(max(values), 6),
                stdev=round(pstdev(values), 6) if len(values) > 1 else 0.0,
            )
        )
    return aggregates
