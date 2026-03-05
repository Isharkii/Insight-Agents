from __future__ import annotations

import json
from datetime import datetime, timezone

from agent.nodes.competitor_node import competitor_node
from app.competitor_intelligence.cache import AsyncTTLCache
from app.competitor_intelligence.schemas import (
    AggregatedMarketMetric,
    CompetitorIntelligenceResponse,
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
