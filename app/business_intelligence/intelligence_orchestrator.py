"""Async intelligence orchestrator that chains context analysis, web search,
news/trends, and signal extraction into a single ``IntelligenceBundle``.

Pipeline stages:
    1. ``ContextAnalyzer``  → structured business context from free text
    2. Search provider       → competitor web search (Brave / Serper / Tavily)
    3. News + Trends         → macro event and keyword signals
    4. Extraction            → deterministic or LLM-structured signal extraction
    5. Aggregation           → unified ``IntelligenceBundle`` with confidence

Every stage is wrapped in a fallback boundary — a single provider failure
degrades the bundle but never crashes the pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, median, pstdev
from typing import Any, List, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.business_intelligence.context_analyzer import (
    BusinessContext,
    ContextAnalyzer,
    ContextAnalyzerError,
)
from app.competitor_intelligence.config import CompetitorIntelligenceConfig
from app.competitor_intelligence.interfaces import Extractor, Scraper, SearchProvider
from app.competitor_intelligence.schemas import (
    ExtractionResult,
    ExtractionSignal,
    ScrapedDocument,
    SearchRequest,
    SearchResponse,
    SourceDocument,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------


class SignalRecord(BaseModel):
    """One intelligence signal with full provenance."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(
        min_length=1, max_length=64,
        description="Origin pipeline stage (e.g. 'search', 'news_api', 'google_trends').",
    )
    metric_name: str = Field(min_length=1, max_length=120)
    value: float
    unit: str = Field(default="ratio", min_length=1, max_length=32)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(default="", max_length=500)


class StageStatus(BaseModel):
    """Execution status of one pipeline stage."""

    model_config = ConfigDict(extra="forbid")

    stage: str = Field(min_length=1, max_length=64)
    status: Literal["success", "degraded", "failed", "skipped"] = "success"
    duration_ms: float = Field(ge=0.0)
    record_count: int = Field(ge=0, default=0)
    error: str | None = None


class AggregatedMetric(BaseModel):
    """Deterministic aggregate across signals sharing the same metric name."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str = Field(min_length=1, max_length=120)
    unit: str = Field(default="ratio", min_length=1, max_length=32)
    sample_size: int = Field(ge=0)
    mean: float | None = None
    median: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    stdev: float | None = None


class IntelligenceBundle(BaseModel):
    """Unified output contract for the intelligence orchestrator.

    Carries the full provenance chain: business context, raw signals,
    deterministic aggregates, pipeline stage statuses, and a composite
    confidence score.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "partial", "failed"]
    generated_at: datetime
    business_context: BusinessContext | None = None
    signals: List[SignalRecord] = Field(default_factory=list)
    aggregated_metrics: List[AggregatedMetric] = Field(default_factory=list)
    stage_statuses: List[StageStatus] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    warnings: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class IntelligenceOrchestrator:
    """Chain context → search → news/trends → extraction → aggregation.

    All external I/O is guarded by per-stage try/except so that a single
    provider outage degrades the bundle rather than raising to the caller.

    Parameters
    ----------
    context_analyzer:
        Synchronous ``ContextAnalyzer`` (runs in executor to avoid blocking).
    search_provider:
        Async web-search provider implementing the ``SearchProvider`` protocol.
    scraper:
        Async URL scraper implementing the ``Scraper`` protocol.
    extractor:
        Async signal extractor implementing the ``Extractor`` protocol.
    news_connector:
        Optional sync news connector (``fetch_records()``).
    trends_connector:
        Optional sync trends connector (``fetch_records()``).
    config:
        ``CompetitorIntelligenceConfig`` (scraper concurrency, query templates …).
    """

    def __init__(
        self,
        *,
        context_analyzer: ContextAnalyzer,
        search_provider: SearchProvider,
        scraper: Scraper,
        extractor: Extractor,
        news_connector: Any | None = None,
        trends_connector: Any | None = None,
        config: CompetitorIntelligenceConfig,
    ) -> None:
        self._context_analyzer = context_analyzer
        self._search_provider = search_provider
        self._scraper = scraper
        self._extractor = extractor
        self._news_connector = news_connector
        self._trends_connector = trends_connector
        self._config = config

    async def run(self, description: str) -> IntelligenceBundle:
        """Execute the full intelligence pipeline.

        Parameters
        ----------
        description:
            Free-text business description, e.g.
            ``"AI SaaS for healthcare clinics in India"``.

        Returns
        -------
        IntelligenceBundle
            Always returns — never raises.  Check ``status`` and
            ``stage_statuses`` for degradation info.
        """
        started_at = time.perf_counter()
        stages: list[StageStatus] = []
        warnings: list[str] = []
        signals: list[SignalRecord] = []

        # -- Stage 1: Context analysis (sync → run in executor) -------------
        context, stage = await self._stage_context(description)
        stages.append(stage)
        if stage.error:
            warnings.append(f"context_analysis: {stage.error}")

        # -- Stages 2-4 in parallel where possible --------------------------
        search_intents = context.search_intents if context else []

        search_task = self._stage_search(search_intents)
        news_task = self._stage_news()
        trends_task = self._stage_trends()

        (search_signals, search_stage), (news_signals, news_stage), (trends_signals, trends_stage) = (
            await asyncio.gather(search_task, news_task, trends_task)
        )

        stages.extend([search_stage, news_stage, trends_stage])
        signals.extend(search_signals)
        signals.extend(news_signals)
        signals.extend(trends_signals)

        for s in [search_stage, news_stage, trends_stage]:
            if s.error:
                warnings.append(f"{s.stage}: {s.error}")

        # -- Stage 5: Aggregate ---------------------------------------------
        aggregated = _aggregate_signals(signals)

        # -- Confidence scoring ---------------------------------------------
        confidence = _compute_bundle_confidence(stages, signals)

        # -- Overall status -------------------------------------------------
        failed_required = any(
            s.status == "failed" for s in stages if s.stage == "context_analysis"
        )
        has_any_failure = any(s.status == "failed" for s in stages)

        if failed_required and not signals:
            status: Literal["success", "partial", "failed"] = "failed"
        elif has_any_failure or any(s.status == "degraded" for s in stages):
            status = "partial"
        else:
            status = "success"

        elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
        logger.info(
            "intelligence_bundle_complete status=%s signals=%d confidence=%.3f elapsed_ms=%s",
            status,
            len(signals),
            confidence,
            elapsed_ms,
        )

        return IntelligenceBundle(
            status=status,
            generated_at=datetime.now(timezone.utc),
            business_context=context,
            signals=signals,
            aggregated_metrics=aggregated,
            stage_statuses=stages,
            confidence=round(confidence, 4),
            warnings=warnings,
        )

    # -----------------------------------------------------------------------
    # Stage implementations
    # -----------------------------------------------------------------------

    async def _stage_context(
        self, description: str,
    ) -> tuple[BusinessContext | None, StageStatus]:
        """Stage 1: Run ContextAnalyzer in an executor (sync → async)."""
        t0 = time.perf_counter()
        try:
            loop = asyncio.get_running_loop()
            context = await loop.run_in_executor(
                None, self._context_analyzer.analyze, description,
            )
            return context, StageStatus(
                stage="context_analysis",
                status="success",
                duration_ms=_elapsed(t0),
                record_count=len(context.search_intents),
            )
        except (ContextAnalyzerError, ValueError) as exc:
            return None, StageStatus(
                stage="context_analysis",
                status="failed",
                duration_ms=_elapsed(t0),
                error=str(exc),
            )

    async def _stage_search(
        self, search_intents: list[str],
    ) -> tuple[list[SignalRecord], StageStatus]:
        """Stage 2: Search → scrape → extract competitor signals."""
        t0 = time.perf_counter()
        if not search_intents:
            return [], StageStatus(
                stage="search_extraction",
                status="skipped",
                duration_ms=_elapsed(t0),
                error="No search intents available (context analysis may have failed).",
            )

        try:
            # 2a. Search across all intents concurrently
            sem = asyncio.Semaphore(max(1, self._config.search_max_concurrency))

            async def _bounded(query: str) -> SearchResponse:
                async with sem:
                    return await self._search_provider.search(
                        SearchRequest(query=query, limit=8),
                    )

            responses: list[SearchResponse] = await asyncio.gather(
                *(_bounded(q) for q in search_intents),
                return_exceptions=False,
            )

            # 2b. Dedupe URLs across all results
            seen_urls: set[str] = set()
            all_docs: list[SourceDocument] = []
            for resp in responses:
                for doc in resp.documents:
                    url_str = str(doc.url)
                    if url_str not in seen_urls:
                        seen_urls.add(url_str)
                        all_docs.append(doc)

            max_pages = self._config.max_pages
            urls_to_scrape = [str(d.url) for d in all_docs[:max_pages]]

            # 2c. Scrape
            scraped: list[ScrapedDocument] = []
            if urls_to_scrape:
                scraped = await self._scraper.fetch_many(
                    urls_to_scrape,
                    max_concurrency=self._config.scraper.max_concurrency,
                )

            # 2d. Extract
            extraction: ExtractionResult = await self._extractor.extract(
                competitor_name="market",
                documents=scraped,
            )

            signals = [
                SignalRecord(
                    source="search",
                    metric_name=sig.metric_name,
                    value=sig.value,
                    unit=sig.unit,
                    confidence=sig.confidence,
                    evidence=sig.evidence,
                )
                for sig in extraction.signals
            ]

            status = "success" if signals else "degraded"
            return signals, StageStatus(
                stage="search_extraction",
                status=status,
                duration_ms=_elapsed(t0),
                record_count=len(signals),
            )

        except Exception as exc:  # noqa: BLE001
            logger.exception("search_extraction_failed error=%s", exc)
            return [], StageStatus(
                stage="search_extraction",
                status="failed",
                duration_ms=_elapsed(t0),
                error=str(exc),
            )

    async def _stage_news(self) -> tuple[list[SignalRecord], StageStatus]:
        """Stage 3a: Fetch news articles via the sync connector."""
        t0 = time.perf_counter()
        if self._news_connector is None:
            return [], StageStatus(
                stage="news",
                status="skipped",
                duration_ms=_elapsed(t0),
                error="No news connector configured.",
            )

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, self._news_connector.fetch_records,
            )
            signals = [
                SignalRecord(
                    source="news_api",
                    metric_name="news_event",
                    value=1.0,
                    unit="count",
                    confidence=0.4,
                    evidence=_extract_title(rec),
                )
                for rec in result.records
            ]

            status = "success" if not result.failed_records else "degraded"
            return signals, StageStatus(
                stage="news",
                status=status,
                duration_ms=_elapsed(t0),
                record_count=len(signals),
            )

        except Exception as exc:  # noqa: BLE001
            logger.exception("news_stage_failed error=%s", exc)
            return [], StageStatus(
                stage="news",
                status="failed",
                duration_ms=_elapsed(t0),
                error=str(exc),
            )

    async def _stage_trends(self) -> tuple[list[SignalRecord], StageStatus]:
        """Stage 3b: Fetch trending keywords via the sync connector."""
        t0 = time.perf_counter()
        if self._trends_connector is None:
            return [], StageStatus(
                stage="trends",
                status="skipped",
                duration_ms=_elapsed(t0),
                error="No trends connector configured.",
            )

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, self._trends_connector.fetch_records,
            )
            signals: list[SignalRecord] = []
            for rec in result.records:
                numeric_value = _safe_float(rec.metric_value)
                if numeric_value is None:
                    continue
                signals.append(
                    SignalRecord(
                        source="google_trends",
                        metric_name="trend_keyword_traffic",
                        value=numeric_value,
                        unit="count",
                        confidence=0.35,
                        evidence=rec.entity_name,
                    ),
                )

            status = "success" if not result.failed_records else "degraded"
            return signals, StageStatus(
                stage="trends",
                status=status,
                duration_ms=_elapsed(t0),
                record_count=len(signals),
            )

        except Exception as exc:  # noqa: BLE001
            logger.exception("trends_stage_failed error=%s", exc)
            return [], StageStatus(
                stage="trends",
                status="failed",
                duration_ms=_elapsed(t0),
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Deterministic helpers (no LLM)
# ---------------------------------------------------------------------------


def _elapsed(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 2)


def _extract_title(record: Any) -> str:
    """Pull a display title from a CanonicalInsightInput's metric_value."""
    mv = record.metric_value
    if isinstance(mv, dict):
        return str(mv.get("title", ""))[:500]
    return str(mv)[:500]


def _safe_float(value: Any) -> float | None:
    """Coerce a value to float; return None if impossible."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _aggregate_signals(signals: Sequence[SignalRecord]) -> list[AggregatedMetric]:
    """Deterministic aggregation — same algorithm as CompetitorIntelligenceService."""
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for sig in signals:
        buckets[(sig.metric_name, sig.unit)].append(sig.value)

    aggregates: list[AggregatedMetric] = []
    for (metric_name, unit), values in sorted(buckets.items()):
        if not values:
            continue
        aggregates.append(
            AggregatedMetric(
                metric_name=metric_name,
                unit=unit,
                sample_size=len(values),
                mean=round(mean(values), 6),
                median=round(median(values), 6),
                min_value=round(min(values), 6),
                max_value=round(max(values), 6),
                stdev=round(pstdev(values), 6) if len(values) > 1 else 0.0,
            ),
        )
    return aggregates


def _compute_bundle_confidence(
    stages: Sequence[StageStatus],
    signals: Sequence[SignalRecord],
) -> float:
    """Deterministic confidence from stage outcomes + signal quality.

    Formula:
        base  = fraction of stages that succeeded or degraded (not failed/skipped)
        depth = bounded signal-count factor  (more signals → higher, capped at 1.0)
        quality = mean signal confidence (or 0 if no signals)

        confidence = 0.4 * base + 0.3 * depth + 0.3 * quality
    """
    if not stages:
        return 0.0

    active_stages = [s for s in stages if s.status != "skipped"]
    if not active_stages:
        return 0.0

    succeeded = sum(1 for s in active_stages if s.status in ("success", "degraded"))
    base = succeeded / len(active_stages)

    depth = min(len(signals) / 10.0, 1.0) if signals else 0.0

    quality = mean(s.confidence for s in signals) if signals else 0.0

    return round(0.4 * base + 0.3 * depth + 0.3 * quality, 4)
