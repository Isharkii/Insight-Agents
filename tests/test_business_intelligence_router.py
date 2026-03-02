"""Tests for POST /api/business-intelligence endpoint."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sys
import types
from pathlib import Path

# The app.api.routers.__init__ transitively imports langgraph (via
# analyze_router).  We pre-seed it as a stub package with the correct
# __path__ so that Python can still locate submodules under it.
import app.api  # noqa: E402 – ensure parent package exists

_routers_dir = str(Path(app.api.__file__).resolve().parent / "routers")
_fake_routers = types.ModuleType("app.api.routers")
_fake_routers.__path__ = [_routers_dir]  # type: ignore[attr-defined]
_fake_routers.__package__ = "app.api.routers"
sys.modules["app.api.routers"] = _fake_routers
# Also set as attribute on parent so unittest.mock.patch can traverse the path
app.api.routers = _fake_routers  # type: ignore[attr-defined]

from app.api.routers.business_intelligence_router import (  # noqa: E402
    BusinessIntelligenceRequest,
    BusinessIntelligenceResponse,
    PipelineStageResult,
    _composite_confidence,
    _elapsed,
    run_business_intelligence,
)

# Register the submodule on the fake package so patch() can find it
import app.api.routers.business_intelligence_router as _bi_mod  # noqa: E402
_fake_routers.business_intelligence_router = _bi_mod  # type: ignore[attr-defined]

from app.business_intelligence.context_analyzer import BusinessContext  # noqa: E402
from app.business_intelligence.insight_synthesizer import (  # noqa: E402
    EmergingSignal,
    InsightBlock,
    SignalReference,
    Zone,
)
from app.business_intelligence.intelligence_orchestrator import (  # noqa: E402
    IntelligenceBundle,
    SignalRecord,
    StageStatus,
)
from app.business_intelligence.strategy_generator import (  # noqa: E402
    CompetitiveAngle,
    RiskMitigation,
    StrategyAction,
    StrategyBlock,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SIGNALS = [
    SignalRecord(source="search", metric_name="listed_price_usd_mean", value=49.0, unit="usd", confidence=0.45, evidence="Detected 1 price mention(s)."),
    SignalRecord(source="search", metric_name="growth_rate_mentioned_pct", value=0.25, unit="ratio", confidence=0.3, evidence="Percent near growth-related keyword."),
    SignalRecord(source="news_api", metric_name="news_event", value=1.0, unit="count", confidence=0.4, evidence="Healthcare AI funding surges"),
    SignalRecord(source="google_trends", metric_name="trend_keyword_traffic", value=50000.0, unit="count", confidence=0.35, evidence="telemedicine"),
    SignalRecord(source="google_trends", metric_name="trend_keyword_traffic", value=12000.0, unit="count", confidence=0.35, evidence="clinic SaaS"),
]

_CONTEXT = BusinessContext(
    industry="healthcare technology",
    business_model="saas",
    target_market="SMB healthcare clinics in India",
    macro_dependencies=["Indian healthcare regulation", "USD/INR exchange rate"],
    search_intents=[
        "AI SaaS healthcare India market size",
        "clinic management software India competitors",
        "ABDM compliance SaaS requirements",
        "healthcare SaaS churn benchmarks",
        "India EHR adoption trends",
    ],
    risk_factors=["Regulatory dependency", "Single-market concentration"],
)

_SIG_REF_0 = SignalReference(signal_id="SIG-0", metric_name="listed_price_usd_mean", value=49.0, unit="usd")
_SIG_REF_1 = SignalReference(signal_id="SIG-1", metric_name="growth_rate_mentioned_pct", value=0.25, unit="ratio")
_SIG_REF_2 = SignalReference(signal_id="SIG-2", metric_name="news_event", value=1.0, unit="count")
_SIG_REF_3 = SignalReference(signal_id="SIG-3", metric_name="trend_keyword_traffic", value=50000.0, unit="count")
_SIG_REF_4 = SignalReference(signal_id="SIG-4", metric_name="trend_keyword_traffic", value=12000.0, unit="count")


def _make_bundle(confidence: float = 0.65) -> IntelligenceBundle:
    return IntelligenceBundle(
        status="partial",
        generated_at=datetime.now(timezone.utc),
        business_context=_CONTEXT,
        signals=_SIGNALS,
        aggregated_metrics=[],
        stage_statuses=[
            StageStatus(stage="context_analysis", status="success", duration_ms=100, record_count=5),
            StageStatus(stage="search_extraction", status="success", duration_ms=200, record_count=2),
            StageStatus(stage="news", status="success", duration_ms=50, record_count=1),
            StageStatus(stage="trends", status="success", duration_ms=40, record_count=2),
        ],
        confidence=confidence,
        warnings=[],
    )


def _make_insight(confidence: float = 0.6) -> InsightBlock:
    return InsightBlock(
        emerging_signals=[
            EmergingSignal(
                title="Healthcare SaaS pricing benchmark at $49/month",
                description="Search extraction identified a market price point of $49/month (SIG-0) suggesting mid-tier positioning.",
                supporting_signals=[_SIG_REF_0],
                relevance="high",
            ),
            EmergingSignal(
                title="Growth rate signal at 25% from market sources",
                description="Market growth rate of 25% (SIG-1) detected from competitor benchmark sources.",
                supporting_signals=[_SIG_REF_1],
                relevance="high",
            ),
            EmergingSignal(
                title="Healthcare AI funding surge detected from news signals",
                description="News signal (SIG-2) shows healthcare AI funding is accelerating globally.",
                supporting_signals=[_SIG_REF_2],
                relevance="medium",
            ),
        ],
        macro_summary=(
            "Healthcare AI funding is accelerating (SIG-2) while telemedicine "
            "search interest shows 50K traffic (SIG-3) and clinic SaaS interest "
            "is at 12K (SIG-4). The macro environment appears supportive."
        ),
        opportunity_zones=[
            Zone(
                title="Telemedicine adoption wave in target market",
                description="High trend traffic for telemedicine (SIG-3) and clinic SaaS (SIG-4) suggests growing demand.",
                supporting_signals=[_SIG_REF_3, _SIG_REF_4],
            ),
        ],
        risk_zones=[
            Zone(
                title="Low confidence in competitor growth signal",
                description="The growth rate signal (SIG-1) carries low extraction confidence (0.3) and may not reflect verified data.",
                supporting_signals=[_SIG_REF_1],
            ),
        ],
        momentum_score=0.6,
        confidence=confidence,
    )


def _make_strategy(confidence: float = 0.55) -> StrategyBlock:
    return StrategyBlock(
        short_term_actions=[
            StrategyAction(
                action="Target $49/month pricing tier based on competitor benchmark signal SIG-0 for healthcare SaaS entry",
                rationale="Market price point of $49/month (SIG-0) suggests mid-tier positioning opportunity for initial launch.",
                supporting_signals=[_SIG_REF_0],
                priority="high",
            ),
            StrategyAction(
                action="Capitalize on 25% sector growth rate (SIG-1) by accelerating product development timeline urgently",
                rationale="Strong growth signal (SIG-1) indicates timing advantage for early market entry.",
                supporting_signals=[_SIG_REF_1],
                priority="high",
            ),
            StrategyAction(
                action="Launch pilot program in telemedicine vertical given strong trend signals SIG-3 and SIG-4",
                rationale="Telemedicine traffic (SIG-3) at 50K and clinic SaaS (SIG-4) at 12K show clear demand.",
                supporting_signals=[_SIG_REF_3, _SIG_REF_4],
                priority="critical",
            ),
        ],
        mid_term_actions=[
            StrategyAction(
                action="Build ABDM compliance module leveraging healthcare AI funding momentum indicated by SIG-2",
                rationale="News signal (SIG-2) on healthcare AI funding suggests regulatory tech investment is timely.",
                supporting_signals=[_SIG_REF_2],
                priority="high",
            ),
            StrategyAction(
                action="Develop competitive pricing tiers around $49 benchmark SIG-0 with premium upsell features",
                rationale="Pricing benchmark (SIG-0) at $49 provides anchor for tiered pricing strategy.",
                supporting_signals=[_SIG_REF_0],
                priority="medium",
            ),
            StrategyAction(
                action="Expand beyond single-market dependency by validating growth signals SIG-1 in adjacent markets",
                rationale="Growth signal (SIG-1) at 25% may indicate broader opportunity beyond India.",
                supporting_signals=[_SIG_REF_1],
                priority="medium",
            ),
        ],
        long_term_positioning=(
            "Position as the leading healthcare SaaS platform for emerging markets, "
            "leveraging telemedicine adoption momentum (SIG-3, SIG-4) and competitive "
            "pricing insights (SIG-0) to capture first-mover advantage in digital "
            "health infrastructure."
        ),
        competitive_angle=CompetitiveAngle(
            positioning="Position against competitors by offering ABDM-compliant SaaS at the market benchmark price (SIG-0) during healthcare AI boom (SIG-2).",
            differentiation="Regulatory compliance combined with competitive pricing (SIG-0) at $49/month creates a defensible differentiation layer.",
            supporting_signals=[_SIG_REF_0, _SIG_REF_2],
        ),
        risk_mitigation=[
            RiskMitigation(
                risk_title="Low confidence in competitor growth signal",
                mitigation="Cross-validate the 25% growth signal (SIG-1) against additional market data sources before major investment decisions.",
                supporting_signals=[_SIG_REF_1],
            ),
        ],
        confidence=confidence,
    )


def _run(coro):
    """Run async coroutine synchronously for testing."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Request schema tests
# ---------------------------------------------------------------------------


class TestRequestSchema:
    def test_valid_request(self) -> None:
        req = BusinessIntelligenceRequest(business_prompt="AI SaaS for healthcare clinics in India")
        assert req.business_prompt == "AI SaaS for healthcare clinics in India"

    def test_rejects_short_prompt(self) -> None:
        with pytest.raises(Exception):
            BusinessIntelligenceRequest(business_prompt="Hi")

    def test_rejects_empty_prompt(self) -> None:
        with pytest.raises(Exception):
            BusinessIntelligenceRequest(business_prompt="")

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(Exception):
            BusinessIntelligenceRequest(business_prompt="AI SaaS for clinics", extra_field="bad")


# ---------------------------------------------------------------------------
# Response schema tests
# ---------------------------------------------------------------------------


class TestResponseSchema:
    def test_valid_response(self) -> None:
        resp = BusinessIntelligenceResponse(
            status="success",
            context=_CONTEXT,
            insights=_make_insight(),
            strategy=_make_strategy(),
            confidence=0.55,
            pipeline=[
                PipelineStageResult(stage="orchestration", status="success", duration_ms=100.0),
            ],
            warnings=[],
            generated_at=datetime.now(timezone.utc),
        )
        assert resp.status == "success"
        assert resp.confidence == 0.55

    def test_partial_response_no_strategy(self) -> None:
        resp = BusinessIntelligenceResponse(
            status="partial",
            context=_CONTEXT,
            insights=_make_insight(),
            strategy=None,
            confidence=0.4,
            pipeline=[],
            warnings=["strategy: failed"],
            generated_at=datetime.now(timezone.utc),
        )
        assert resp.strategy is None

    def test_failed_response(self) -> None:
        resp = BusinessIntelligenceResponse(
            status="failed",
            context=None,
            insights=None,
            strategy=None,
            confidence=0.0,
            pipeline=[],
            warnings=["orchestration: boom"],
            generated_at=datetime.now(timezone.utc),
        )
        assert resp.confidence == 0.0


# ---------------------------------------------------------------------------
# Pipeline stage result tests
# ---------------------------------------------------------------------------


class TestPipelineStageResult:
    def test_success_stage(self) -> None:
        stage = PipelineStageResult(stage="orchestration", status="success", duration_ms=123.45)
        assert stage.error is None

    def test_failed_stage_with_error(self) -> None:
        stage = PipelineStageResult(stage="synthesis", status="failed", duration_ms=50.0, error="boom")
        assert stage.error == "boom"

    def test_skipped_stage(self) -> None:
        stage = PipelineStageResult(stage="strategy", status="skipped", duration_ms=0.1, error="No data")
        assert stage.status == "skipped"


# ---------------------------------------------------------------------------
# Composite confidence tests
# ---------------------------------------------------------------------------


class TestCompositeConfidence:
    def test_all_present(self) -> None:
        bundle = _make_bundle(confidence=0.65)
        insight = _make_insight(confidence=0.6)
        strategy = _make_strategy(confidence=0.55)
        score = _composite_confidence(bundle, insight, strategy)
        # 0.65*0.4 + 0.6*0.35 + 0.55*0.25 = 0.26 + 0.21 + 0.1375 = 0.6075
        assert abs(score - 0.6075) < 1e-4

    def test_bundle_only(self) -> None:
        bundle = _make_bundle(confidence=0.65)
        score = _composite_confidence(bundle, None, None)
        assert abs(score - 0.65) < 1e-4

    def test_bundle_and_insight(self) -> None:
        bundle = _make_bundle(confidence=0.65)
        insight = _make_insight(confidence=0.6)
        score = _composite_confidence(bundle, insight, None)
        # (0.65*0.4 + 0.6*0.35) / (0.4 + 0.35) = (0.26 + 0.21) / 0.75 = 0.6267
        assert abs(score - 0.6267) < 1e-3

    def test_all_none(self) -> None:
        score = _composite_confidence(None, None, None)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Elapsed helper tests
# ---------------------------------------------------------------------------


class TestElapsed:
    def test_returns_positive_float(self) -> None:
        import time
        t0 = time.perf_counter()
        time.sleep(0.01)
        result = _elapsed(t0)
        assert result > 0.0
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Endpoint integration tests (mocked services)
# ---------------------------------------------------------------------------


class TestEndpointHappyPath:
    """Full pipeline success with mocked services."""

    def test_full_success(self) -> None:
        bundle = _make_bundle(confidence=0.65)
        insight = _make_insight(confidence=0.6)
        strategy = _make_strategy(confidence=0.55)

        with patch("app.api.routers.business_intelligence_router._build_llm_adapter") as mock_adapter, \
             patch("app.api.routers.business_intelligence_router._build_orchestrator_deps") as mock_deps, \
             patch("app.api.routers.business_intelligence_router.IntelligenceOrchestrator") as mock_orch_cls, \
             patch("app.api.routers.business_intelligence_router.InsightSynthesizer") as mock_synth_cls, \
             patch("app.api.routers.business_intelligence_router.StrategyGenerator") as mock_strat_cls, \
             patch("app.api.routers.business_intelligence_router.ContextAnalyzer"):

            mock_adapter.return_value = MagicMock()
            mock_deps.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock(), None, None)

            mock_orch = MagicMock()
            mock_orch.run = AsyncMock(return_value=bundle)
            mock_orch_cls.return_value = mock_orch

            mock_synth = MagicMock()
            mock_synth.synthesize.return_value = insight
            mock_synth_cls.return_value = mock_synth

            mock_strat = MagicMock()
            mock_strat.generate.return_value = strategy
            mock_strat_cls.return_value = mock_strat

            body = BusinessIntelligenceRequest(business_prompt="AI SaaS for healthcare clinics in India")
            resp = _run(run_business_intelligence(body))

            assert resp.status == "success"
            assert resp.context == _CONTEXT
            assert resp.insights is not None
            assert resp.strategy is not None
            assert resp.confidence > 0.0
            assert len(resp.pipeline) == 3
            assert all(s.status == "success" for s in resp.pipeline)
            assert isinstance(resp.generated_at, datetime)


class TestEndpointOrchestrationFailure:
    """Orchestration fails → synthesis/strategy skipped → partial/failed status."""

    def test_orchestration_exception(self) -> None:
        with patch("app.api.routers.business_intelligence_router._build_llm_adapter") as mock_adapter, \
             patch("app.api.routers.business_intelligence_router._build_orchestrator_deps") as mock_deps, \
             patch("app.api.routers.business_intelligence_router.IntelligenceOrchestrator") as mock_orch_cls, \
             patch("app.api.routers.business_intelligence_router.InsightSynthesizer") as mock_synth_cls, \
             patch("app.api.routers.business_intelligence_router.StrategyGenerator") as mock_strat_cls, \
             patch("app.api.routers.business_intelligence_router.ContextAnalyzer"):

            mock_adapter.return_value = MagicMock()
            mock_deps.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock(), None, None)

            mock_orch = MagicMock()
            mock_orch.run = AsyncMock(side_effect=RuntimeError("Search provider down"))
            mock_orch_cls.return_value = mock_orch
            mock_synth_cls.return_value = MagicMock()
            mock_strat_cls.return_value = MagicMock()

            body = BusinessIntelligenceRequest(business_prompt="AI SaaS for healthcare clinics in India")
            resp = _run(run_business_intelligence(body))

            assert resp.status == "failed"
            assert resp.context is None
            assert resp.insights is None
            assert resp.strategy is None
            assert resp.confidence == 0.0
            assert any("orchestration" in s.stage for s in resp.pipeline)
            assert any("Search provider down" in w for w in resp.warnings)


class TestEndpointSynthesisFailure:
    """Orchestration succeeds, synthesis fails → partial status."""

    def test_synthesis_failure_returns_partial(self) -> None:
        bundle = _make_bundle(confidence=0.65)

        with patch("app.api.routers.business_intelligence_router._build_llm_adapter") as mock_adapter, \
             patch("app.api.routers.business_intelligence_router._build_orchestrator_deps") as mock_deps, \
             patch("app.api.routers.business_intelligence_router.IntelligenceOrchestrator") as mock_orch_cls, \
             patch("app.api.routers.business_intelligence_router.InsightSynthesizer") as mock_synth_cls, \
             patch("app.api.routers.business_intelligence_router.StrategyGenerator") as mock_strat_cls, \
             patch("app.api.routers.business_intelligence_router.ContextAnalyzer"):

            mock_adapter.return_value = MagicMock()
            mock_deps.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock(), None, None)

            mock_orch = MagicMock()
            mock_orch.run = AsyncMock(return_value=bundle)
            mock_orch_cls.return_value = mock_orch

            mock_synth = MagicMock()
            mock_synth.synthesize.side_effect = ValueError("LLM returned garbage")
            mock_synth_cls.return_value = mock_synth

            mock_strat_cls.return_value = MagicMock()

            body = BusinessIntelligenceRequest(business_prompt="AI SaaS for healthcare clinics in India")
            resp = _run(run_business_intelligence(body))

            assert resp.status == "partial"
            assert resp.context is not None  # from bundle
            assert resp.insights is None
            assert resp.strategy is None
            assert resp.confidence > 0.0  # bundle contributed
            assert any("synthesis" in w for w in resp.warnings)


class TestEndpointStrategyFailure:
    """Orchestration + synthesis succeed, strategy fails → partial status."""

    def test_strategy_failure_returns_partial(self) -> None:
        bundle = _make_bundle(confidence=0.65)
        insight = _make_insight(confidence=0.6)

        with patch("app.api.routers.business_intelligence_router._build_llm_adapter") as mock_adapter, \
             patch("app.api.routers.business_intelligence_router._build_orchestrator_deps") as mock_deps, \
             patch("app.api.routers.business_intelligence_router.IntelligenceOrchestrator") as mock_orch_cls, \
             patch("app.api.routers.business_intelligence_router.InsightSynthesizer") as mock_synth_cls, \
             patch("app.api.routers.business_intelligence_router.StrategyGenerator") as mock_strat_cls, \
             patch("app.api.routers.business_intelligence_router.ContextAnalyzer"):

            mock_adapter.return_value = MagicMock()
            mock_deps.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock(), None, None)

            mock_orch = MagicMock()
            mock_orch.run = AsyncMock(return_value=bundle)
            mock_orch_cls.return_value = mock_orch

            mock_synth = MagicMock()
            mock_synth.synthesize.return_value = insight
            mock_synth_cls.return_value = mock_synth

            mock_strat = MagicMock()
            mock_strat.generate.side_effect = ValueError("Generic advice rejected")
            mock_strat_cls.return_value = mock_strat

            body = BusinessIntelligenceRequest(business_prompt="AI SaaS for healthcare clinics in India")
            resp = _run(run_business_intelligence(body))

            assert resp.status == "partial"
            assert resp.insights is not None
            assert resp.strategy is None
            assert any("strategy" in w for w in resp.warnings)


class TestEndpointServiceInitFailure:
    """Service construction failure raises HTTPException."""

    def test_raises_500_on_init_failure(self) -> None:
        from fastapi import HTTPException

        with patch("app.api.routers.business_intelligence_router._build_llm_adapter", side_effect=RuntimeError("No API key")):
            body = BusinessIntelligenceRequest(business_prompt="AI SaaS for healthcare clinics in India")
            with pytest.raises(HTTPException) as exc_info:
                _run(run_business_intelligence(body))
            assert exc_info.value.status_code == 500


class TestEndpointEmptySignals:
    """Orchestrator returns bundle with no signals → synthesis skipped."""

    def test_empty_signals_skips_synthesis(self) -> None:
        bundle = IntelligenceBundle(
            status="partial",
            generated_at=datetime.now(timezone.utc),
            business_context=_CONTEXT,
            signals=[],
            aggregated_metrics=[],
            stage_statuses=[
                StageStatus(stage="context_analysis", status="success", duration_ms=100, record_count=0),
            ],
            confidence=0.2,
            warnings=["No signals extracted"],
        )

        with patch("app.api.routers.business_intelligence_router._build_llm_adapter") as mock_adapter, \
             patch("app.api.routers.business_intelligence_router._build_orchestrator_deps") as mock_deps, \
             patch("app.api.routers.business_intelligence_router.IntelligenceOrchestrator") as mock_orch_cls, \
             patch("app.api.routers.business_intelligence_router.InsightSynthesizer") as mock_synth_cls, \
             patch("app.api.routers.business_intelligence_router.StrategyGenerator") as mock_strat_cls, \
             patch("app.api.routers.business_intelligence_router.ContextAnalyzer"):

            mock_adapter.return_value = MagicMock()
            mock_deps.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock(), None, None)

            mock_orch = MagicMock()
            mock_orch.run = AsyncMock(return_value=bundle)
            mock_orch_cls.return_value = mock_orch
            mock_synth_cls.return_value = MagicMock()
            mock_strat_cls.return_value = MagicMock()

            body = BusinessIntelligenceRequest(business_prompt="AI SaaS for healthcare clinics in India")
            resp = _run(run_business_intelligence(body))

            assert resp.insights is None
            assert resp.strategy is None
            # Synthesis and strategy should be skipped
            synth_stage = next(s for s in resp.pipeline if s.stage == "synthesis")
            assert synth_stage.status == "skipped"
            strat_stage = next(s for s in resp.pipeline if s.stage == "strategy")
            assert strat_stage.status == "skipped"
