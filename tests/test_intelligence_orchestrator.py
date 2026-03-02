"""Tests for IntelligenceOrchestrator — async pipeline with fallback."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest

from app.business_intelligence.context_analyzer import (
    BusinessContext,
    ContextAnalyzer,
)
from app.business_intelligence.intelligence_orchestrator import (
    IntelligenceBundle,
    IntelligenceOrchestrator,
    SignalRecord,
    StageStatus,
    _aggregate_signals,
    _compute_bundle_confidence,
)
from app.competitor_intelligence.config import CompetitorIntelligenceConfig
from app.competitor_intelligence.schemas import (
    ExtractionResult,
    ExtractionSignal,
    ScrapedDocument,
    SearchRequest,
    SearchResponse,
    SourceDocument,
)
from llm_synthesis.adapter import BaseLLMAdapter


# ---------------------------------------------------------------------------
# Shared mock data
# ---------------------------------------------------------------------------

_VALID_CONTEXT_JSON = json.dumps({
    "industry": "healthcare technology",
    "business_model": "saas",
    "target_market": "SMB healthcare clinics in India",
    "macro_dependencies": [
        "Indian healthcare regulation",
        "USD/INR exchange rate",
    ],
    "search_intents": [
        "AI SaaS healthcare India market size",
        "clinic management software India competitors",
        "ABDM compliance SaaS requirements",
        "healthcare SaaS churn benchmarks",
        "India EHR adoption trends",
    ],
    "risk_factors": [
        "Regulatory dependency",
        "Single-market concentration",
    ],
})


# ---------------------------------------------------------------------------
# Mock adapters / providers
# ---------------------------------------------------------------------------


class _MockLLMAdapter(BaseLLMAdapter):
    def generate(self, prompt: str) -> str:
        return _VALID_CONTEXT_JSON


class _FailingLLMAdapter(BaseLLMAdapter):
    def generate(self, prompt: str) -> str:
        return "NOT JSON"


class _MockSearchProvider:
    name = "mock_search"

    async def search(self, request: SearchRequest) -> SearchResponse:
        return SearchResponse(
            provider="mock_search",
            query=request.query,
            fetched_at=datetime.now(timezone.utc),
            documents=[
                SourceDocument(
                    provider="mock_search",
                    rank=1,
                    url="https://example.com/article-1",
                    title="Mock article",
                    snippet="Growth rate is 25%",
                    domain="example.com",
                ),
            ],
        )


class _FailingSearchProvider:
    name = "failing_search"

    async def search(self, request: SearchRequest) -> SearchResponse:
        raise ConnectionError("Search provider down")


class _MockScraper:
    async def fetch(self, url: str) -> ScrapedDocument:
        return ScrapedDocument(
            url=url,
            fetched_at=datetime.now(timezone.utc),
            status_code=200,
            title="Mock page",
            text="Our growth rate is 25% and pricing starts at $49/month. Churn is 5%.",
        )

    async def fetch_many(
        self, urls: Sequence[str], *, max_concurrency: int,
    ) -> list[ScrapedDocument]:
        return [await self.fetch(url) for url in urls]


class _MockExtractor:
    async def extract(
        self, *, competitor_name: str, documents: Sequence[ScrapedDocument],
    ) -> ExtractionResult:
        return ExtractionResult(
            competitor_name=competitor_name,
            extraction_method="deterministic",
            extracted_at=datetime.now(timezone.utc),
            signals=[
                ExtractionSignal(
                    metric_name="listed_price_usd_mean",
                    value=49.0,
                    unit="usd",
                    signal_type="competitor_metric",
                    confidence=0.45,
                    source_url="https://example.com/article-1",
                    evidence="Detected 1 price mention(s).",
                ),
                ExtractionSignal(
                    metric_name="growth_rate_mentioned_pct",
                    value=0.25,
                    unit="ratio",
                    signal_type="industry_metric",
                    confidence=0.3,
                    source_url="https://example.com/article-1",
                    evidence="Percent near growth-related keyword.",
                ),
            ],
        )


@dataclass
class _FakeCanonicalRecord:
    entity_name: str
    metric_value: Any


@dataclass
class _FakeConnectorResult:
    source: str
    records: list
    failed_records: int = 0


class _MockNewsConnector:
    source = "news_api"

    def fetch_records(self) -> _FakeConnectorResult:
        return _FakeConnectorResult(
            source="news_api",
            records=[
                _FakeCanonicalRecord(
                    entity_name="Reuters",
                    metric_value={"title": "Healthcare AI funding surges"},
                ),
            ],
        )


class _MockTrendsConnector:
    source = "google_trends"

    def fetch_records(self) -> _FakeConnectorResult:
        return _FakeConnectorResult(
            source="google_trends",
            records=[
                _FakeCanonicalRecord(entity_name="telemedicine", metric_value=50000),
                _FakeCanonicalRecord(entity_name="clinic SaaS", metric_value=12000),
            ],
        )


class _FailingNewsConnector:
    source = "news_api"

    def fetch_records(self):
        raise ConnectionError("News API unreachable")


def _build_config() -> CompetitorIntelligenceConfig:
    return CompetitorIntelligenceConfig(
        provider="brave",
        max_pages=2,
        search_max_concurrency=4,
    )


def _build_orchestrator(**overrides: Any) -> IntelligenceOrchestrator:
    defaults = dict(
        context_analyzer=ContextAnalyzer(_MockLLMAdapter()),
        search_provider=_MockSearchProvider(),
        scraper=_MockScraper(),
        extractor=_MockExtractor(),
        news_connector=_MockNewsConnector(),
        trends_connector=_MockTrendsConnector(),
        config=_build_config(),
    )
    defaults.update(overrides)
    return IntelligenceOrchestrator(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIntelligenceBundleSchema:
    def test_minimal_valid(self) -> None:
        bundle = IntelligenceBundle(
            status="success",
            generated_at=datetime.now(timezone.utc),
            confidence=0.5,
        )
        assert bundle.status == "success"
        assert bundle.signals == []

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(Exception):
            IntelligenceBundle(
                status="success",
                generated_at=datetime.now(timezone.utc),
                confidence=0.5,
                revenue=100,  # type: ignore[call-arg]
            )


def _run(coro):
    """Run an async coroutine synchronously — no plugin needed."""
    return asyncio.run(coro)


class TestOrchestratorHappyPath:
    def test_full_pipeline(self) -> None:
        orch = _build_orchestrator()
        bundle = _run(orch.run("AI SaaS for healthcare clinics in India"))

        assert bundle.status in ("success", "partial")
        assert bundle.business_context is not None
        assert bundle.business_context.business_model == "saas"
        assert len(bundle.signals) > 0
        assert bundle.confidence > 0.0

        stage_names = {s.stage for s in bundle.stage_statuses}
        assert stage_names == {"context_analysis", "search_extraction", "news", "trends"}

    def test_signals_have_source(self) -> None:
        orch = _build_orchestrator()
        bundle = _run(orch.run("AI SaaS for healthcare clinics in India"))

        sources = {s.source for s in bundle.signals}
        assert "search" in sources
        assert "news_api" in sources
        assert "google_trends" in sources

    def test_aggregated_metrics(self) -> None:
        orch = _build_orchestrator()
        bundle = _run(orch.run("AI SaaS for healthcare clinics in India"))

        metric_names = {m.metric_name for m in bundle.aggregated_metrics}
        assert "listed_price_usd_mean" in metric_names


class TestOrchestratorFallbacks:
    def test_search_failure_degrades(self) -> None:
        orch = _build_orchestrator(search_provider=_FailingSearchProvider())
        bundle = _run(orch.run("AI SaaS for healthcare clinics in India"))

        assert bundle.status == "partial"
        search_stage = next(s for s in bundle.stage_statuses if s.stage == "search_extraction")
        assert search_stage.status == "failed"
        # News + trends still produce signals
        assert len(bundle.signals) > 0

    def test_news_failure_degrades(self) -> None:
        orch = _build_orchestrator(news_connector=_FailingNewsConnector())
        bundle = _run(orch.run("AI SaaS for healthcare clinics in India"))

        assert bundle.status == "partial"
        news_stage = next(s for s in bundle.stage_statuses if s.stage == "news")
        assert news_stage.status == "failed"

    def test_no_connectors_skips_gracefully(self) -> None:
        orch = _build_orchestrator(news_connector=None, trends_connector=None)
        bundle = _run(orch.run("AI SaaS for healthcare clinics in India"))

        news_stage = next(s for s in bundle.stage_statuses if s.stage == "news")
        trends_stage = next(s for s in bundle.stage_statuses if s.stage == "trends")
        assert news_stage.status == "skipped"
        assert trends_stage.status == "skipped"

    def test_context_failure_skips_search(self) -> None:
        orch = _build_orchestrator(
            context_analyzer=ContextAnalyzer(_FailingLLMAdapter(), max_retries=0),
        )
        bundle = _run(orch.run("anything"))

        assert bundle.business_context is None
        ctx_stage = next(s for s in bundle.stage_statuses if s.stage == "context_analysis")
        assert ctx_stage.status == "failed"
        search_stage = next(s for s in bundle.stage_statuses if s.stage == "search_extraction")
        assert search_stage.status == "skipped"


class TestConfidenceScoring:
    def test_all_success_high_signal_count(self) -> None:
        stages = [
            StageStatus(stage="a", status="success", duration_ms=10, record_count=5),
            StageStatus(stage="b", status="success", duration_ms=10, record_count=5),
        ]
        signals = [
            SignalRecord(source="x", metric_name="m", value=1.0, confidence=0.8)
            for _ in range(10)
        ]
        conf = _compute_bundle_confidence(stages, signals)
        # base=1.0, depth=1.0, quality=0.8 → 0.4 + 0.3 + 0.24 = 0.94
        assert conf == pytest.approx(0.94, abs=0.01)

    def test_all_failed_zero_signals(self) -> None:
        stages = [
            StageStatus(stage="a", status="failed", duration_ms=10),
        ]
        conf = _compute_bundle_confidence(stages, [])
        assert conf == 0.0

    def test_skipped_stages_excluded(self) -> None:
        stages = [
            StageStatus(stage="a", status="success", duration_ms=10),
            StageStatus(stage="b", status="skipped", duration_ms=0),
        ]
        signals = [
            SignalRecord(source="x", metric_name="m", value=1.0, confidence=0.5)
        ]
        conf = _compute_bundle_confidence(stages, signals)
        # base=1.0, depth=0.1, quality=0.5 → 0.4 + 0.03 + 0.15 = 0.58
        assert conf == pytest.approx(0.58, abs=0.01)


class TestAggregation:
    def test_aggregates_by_metric_name(self) -> None:
        signals = [
            SignalRecord(source="a", metric_name="price", value=10.0, unit="usd", confidence=0.5),
            SignalRecord(source="b", metric_name="price", value=20.0, unit="usd", confidence=0.5),
            SignalRecord(source="c", metric_name="growth", value=0.3, unit="ratio", confidence=0.4),
        ]
        aggs = _aggregate_signals(signals)
        assert len(aggs) == 2

        price_agg = next(a for a in aggs if a.metric_name == "price")
        assert price_agg.sample_size == 2
        assert price_agg.mean == pytest.approx(15.0)
        assert price_agg.min_value == pytest.approx(10.0)
        assert price_agg.max_value == pytest.approx(20.0)

    def test_empty_signals(self) -> None:
        assert _aggregate_signals([]) == []
