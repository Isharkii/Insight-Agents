from __future__ import annotations

import json
from datetime import datetime, timezone

from agent.nodes.competitor_node import competitor_node
from app.competitor_intelligence.cache import AsyncTTLCache
from app.competitor_intelligence.schemas import (
    AggregatedMarketMetric,
    CompetitorProfile,
    CompetitorIntelligenceResponse,
    ExtractionResult,
    SourceDocument,
)


class _FakeService:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, request):  # noqa: ANN001
        self.calls += 1
        return CompetitorIntelligenceResponse(
            status="success",
            generated_at=datetime.now(timezone.utc),
            subject_entity=request.subject_entity,
            competitor_profiles=[],
            aggregated_market_data=[
                AggregatedMarketMetric(
                    metric_name="listed_price_usd_mean",
                    unit="usd",
                    sample_size=3,
                    mean=49.0,
                    median=49.0,
                    min_value=39.0,
                    max_value=59.0,
                    stdev=8.2,
                ),
            ],
            warnings=["partial_source_coverage"],
        )


class _FakeServiceWithNews:
    async def generate(self, request):  # noqa: ANN001
        now = datetime.now(timezone.utc)
        return CompetitorIntelligenceResponse(
            status="success",
            generated_at=now,
            subject_entity=request.subject_entity,
            competitor_profiles=[
                CompetitorProfile(
                    competitor_name="Slack",
                    queries=["Slack recent news"],
                    search_documents=[
                        SourceDocument(
                            provider="brave",
                            rank=1,
                            url="https://example.com/slack-lawsuit",
                            title="Slack faces antitrust lawsuit in key market",
                            snippet="Regulatory complaint may increase pricing and retention risk.",
                            published_at=now,
                            domain="example.com",
                        )
                    ],
                    scraped_documents=[],
                    extraction=ExtractionResult(
                        competitor_name="Slack",
                        extraction_method="deterministic",
                        extracted_at=now,
                        signals=[],
                        warnings=[],
                    ),
                    warnings=[],
                )
            ],
            aggregated_market_data=[
                AggregatedMarketMetric(
                    metric_name="listed_price_usd_mean",
                    unit="usd",
                    sample_size=3,
                    mean=49.0,
                    median=49.0,
                    min_value=39.0,
                    max_value=59.0,
                    stdev=8.2,
                ),
            ],
            warnings=[],
        )


def _base_state() -> dict:
    return {
        "business_type": "saas",
        "entity_name": "Acme",
        "competitive_context": {
            "available": True,
            "peer_count": 2,
            "peers": ["Contoso", "Globex"],
            "metrics": ["mrr", "churn_rate"],
            "benchmark_rows_count": 18,
        },
    }


def test_competitor_node_disabled_preserves_deterministic_mode(monkeypatch) -> None:
    monkeypatch.setenv("COMP_INTEL_ANALYZE_ENABLED", "false")
    state = _base_state()

    result = competitor_node(state)
    ctx = result["competitive_context"]

    assert ctx["source"] == "deterministic_local"
    assert ctx["available"] is True
    assert ctx["peer_count"] == 2
    assert ctx["numeric_signals"] == []


def test_competitor_node_external_fetch_uses_24h_cache_and_numeric_contract(monkeypatch) -> None:
    monkeypatch.setenv("COMP_INTEL_ANALYZE_ENABLED", "true")
    fake_service = _FakeService()

    monkeypatch.setattr(
        "agent.nodes.competitor_node._build_service_singleton",
        lambda: fake_service,
    )
    monkeypatch.setattr(
        "agent.nodes.competitor_node._COMP_CONTEXT_CACHE",
        AsyncTTLCache(ttl_seconds=24 * 60 * 60, max_size=16),
    )

    first = competitor_node(_base_state())
    second = competitor_node(_base_state())

    assert fake_service.calls == 1

    ctx = first["competitive_context"]
    assert ctx["source"] == "external_fetch"
    assert ctx["available"] is True
    assert ctx["numeric_signals"]

    metric = ctx["numeric_signals"][0]
    assert metric["metric_name"] == "listed_price_usd_mean"
    assert metric["unit"] == "usd"
    assert metric["sample_size"] == 3

    cached_ctx = second["competitive_context"]
    assert cached_ctx["cache_hit"] is True

    serialized = json.dumps(ctx)
    assert "search_documents" not in serialized
    assert "scraped_documents" not in serialized
    assert "evidence" not in serialized


def test_competitor_node_surfaces_critical_news_highlights(monkeypatch) -> None:
    monkeypatch.setenv("COMP_INTEL_ANALYZE_ENABLED", "true")
    monkeypatch.setattr(
        "agent.nodes.competitor_node._build_service_singleton",
        lambda: _FakeServiceWithNews(),
    )
    monkeypatch.setattr(
        "agent.nodes.competitor_node._COMP_CONTEXT_CACHE",
        AsyncTTLCache(ttl_seconds=24 * 60 * 60, max_size=16),
    )

    result = competitor_node(_base_state())
    ctx = result["competitive_context"]
    news = ctx.get("news_highlights") or []

    assert isinstance(news, list)
    assert len(news) >= 1
    assert news[0]["competitor"] == "Slack"
    assert "lawsuit" in news[0]["title"].lower()
    assert int(news[0].get("criticality_score", 0)) >= 1
