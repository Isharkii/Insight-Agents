"""Tests for StrategyGenerator — schema, validation, prompt, fallback."""

from __future__ import annotations

import json
from typing import List

import pytest

from app.business_intelligence.insight_synthesizer import (
    EmergingSignal,
    InsightBlock,
    SignalReference,
    Zone,
)
from app.business_intelligence.strategy_generator import (
    CompetitiveAngle,
    RiskMitigation,
    StrategyAction,
    StrategyBlock,
    StrategyGenerator,
    StrategyGeneratorError,
    _collect_signal_refs,
    _validate_confidence_ceiling,
    _validate_signal_ids_in_scope,
)
from llm_synthesis.adapter import BaseLLMAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_REFS = {
    "price": SignalReference(signal_id="SIG-0", metric_name="listed_price_usd_mean", value=49.0, unit="usd"),
    "growth": SignalReference(signal_id="SIG-1", metric_name="growth_rate_mentioned_pct", value=0.25, unit="ratio"),
    "news": SignalReference(signal_id="SIG-2", metric_name="news_event", value=1.0, unit="count"),
    "trend1": SignalReference(signal_id="SIG-3", metric_name="trend_keyword_traffic", value=50000.0, unit="count"),
    "trend2": SignalReference(signal_id="SIG-4", metric_name="trend_keyword_traffic", value=12000.0, unit="count"),
}


def _make_insight(confidence: float = 0.65) -> InsightBlock:
    return InsightBlock(
        emerging_signals=[
            EmergingSignal(
                title="Healthcare SaaS pricing benchmark at $49/month",
                description="Search extraction identified a market price point of $49/month (SIG-0) suggesting mid-tier positioning.",
                supporting_signals=[_REFS["price"]],
                relevance="high",
            ),
            EmergingSignal(
                title="Growth rate signal at 25% from market sources",
                description="Market growth rate of 25% (SIG-1) detected from competitor benchmark sources indicating strong momentum.",
                supporting_signals=[_REFS["growth"]],
                relevance="high",
            ),
            EmergingSignal(
                title="Healthcare AI funding acceleration reported by news sources",
                description="News signal (SIG-2) confirms accelerating funding into healthcare AI startups and digital health platforms.",
                supporting_signals=[_REFS["news"]],
                relevance="medium",
            ),
        ],
        macro_summary=(
            "Healthcare AI funding is accelerating (SIG-2) while telemedicine "
            "search interest shows 50K traffic (SIG-3) and clinic SaaS interest "
            "is at 12K (SIG-4)."
        ),
        opportunity_zones=[
            Zone(
                title="Telemedicine adoption wave in target market",
                description="High trend traffic for telemedicine (SIG-3) and clinic SaaS (SIG-4) suggests growing demand.",
                supporting_signals=[_REFS["trend1"], _REFS["trend2"]],
            ),
        ],
        risk_zones=[
            Zone(
                title="Low confidence in competitor growth signal",
                description="Growth rate signal (SIG-1) carries low extraction confidence (0.3) and may not reflect verified data.",
                supporting_signals=[_REFS["growth"]],
            ),
        ],
        momentum_score=0.6,
        confidence=confidence,
    )


def _valid_strategy_dict(confidence: float = 0.6) -> dict:
    return {
        "short_term_actions": [
            {
                "action": "Set introductory pricing at $39/month to undercut competitor benchmark of $49 (SIG-0) and capture early-mover clinics.",
                "rationale": "Market pricing signal SIG-0 shows $49 median — a $39 entry point creates competitive gap for acquisition.",
                "supporting_signals": [
                    {"signal_id": "SIG-0", "metric_name": "listed_price_usd_mean", "value": 49.0, "unit": "usd"},
                ],
                "priority": "critical",
            },
            {
                "action": "Launch telemedicine integration feature to ride the 50K search-traffic wave identified in SIG-3.",
                "rationale": "Trend signal SIG-3 shows 50K searches for telemedicine — aligning product to this demand captures organic interest.",
                "supporting_signals": [
                    {"signal_id": "SIG-3", "metric_name": "trend_keyword_traffic", "value": 50000.0, "unit": "count"},
                ],
                "priority": "high",
            },
            {
                "action": "Publish healthcare AI funding analysis using news signal SIG-2 to build thought leadership and inbound pipeline.",
                "rationale": "News signal SIG-2 confirms funding acceleration — content marketing anchored to this trend drives awareness.",
                "supporting_signals": [
                    {"signal_id": "SIG-2", "metric_name": "news_event", "value": 1.0, "unit": "count"},
                ],
                "priority": "medium",
            },
        ],
        "mid_term_actions": [
            {
                "action": "Build clinic-SaaS vertical partnerships leveraging 12K search interest (SIG-4) to create distribution channels.",
                "rationale": "Trend signal SIG-4 validates clinic SaaS demand — partnerships convert this into scalable distribution.",
                "supporting_signals": [
                    {"signal_id": "SIG-4", "metric_name": "trend_keyword_traffic", "value": 12000.0, "unit": "count"},
                ],
                "priority": "high",
            },
            {
                "action": "Develop competitor growth tracking dashboard to validate the 25% growth signal (SIG-1) before committing resources.",
                "rationale": "Growth signal SIG-1 has low extraction confidence — building internal tracking reduces dependency on scraped estimates.",
                "supporting_signals": [
                    {"signal_id": "SIG-1", "metric_name": "growth_rate_mentioned_pct", "value": 0.25, "unit": "ratio"},
                ],
                "priority": "high",
            },
            {
                "action": "Expand product coverage to capture telemedicine + clinic SaaS intersection per trend signals SIG-3 and SIG-4.",
                "rationale": "Combined trend signals SIG-3 (50K) and SIG-4 (12K) point to a converging demand corridor worth building into.",
                "supporting_signals": [
                    {"signal_id": "SIG-3", "metric_name": "trend_keyword_traffic", "value": 50000.0, "unit": "count"},
                    {"signal_id": "SIG-4", "metric_name": "trend_keyword_traffic", "value": 12000.0, "unit": "count"},
                ],
                "priority": "medium",
            },
        ],
        "long_term_positioning": (
            "Position as the integrated healthcare intelligence platform for "
            "emerging-market clinics.  The $49 price benchmark (SIG-0), 25% "
            "market growth (SIG-1), and rising telemedicine demand (SIG-3) "
            "indicate a 12+ month runway to establish category leadership "
            "before competitor density increases."
        ),
        "competitive_angle": {
            "positioning": "Undercut competitor pricing anchor of $49 (SIG-0) while offering telemedicine-native workflow absent from incumbents.",
            "differentiation": "Combine healthcare AI funding trend (SIG-2) with telemedicine demand (SIG-3) to build a category incumbents cannot easily replicate.",
            "supporting_signals": [
                {"signal_id": "SIG-0", "metric_name": "listed_price_usd_mean", "value": 49.0, "unit": "usd"},
                {"signal_id": "SIG-3", "metric_name": "trend_keyword_traffic", "value": 50000.0, "unit": "count"},
            ],
        },
        "risk_mitigation": [
            {
                "risk_title": "Low confidence in competitor growth signal",
                "mitigation": "Cross-validate the 25% growth signal (SIG-1) with proprietary data and industry reports before using it in forecasting models.",
                "supporting_signals": [
                    {"signal_id": "SIG-1", "metric_name": "growth_rate_mentioned_pct", "value": 0.25, "unit": "ratio"},
                ],
            },
        ],
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Mock adapters
# ---------------------------------------------------------------------------


class _MockStrategyAdapter(BaseLLMAdapter):
    def __init__(self, confidence: float = 0.6) -> None:
        self._confidence = confidence

    def generate(self, prompt: str) -> str:
        return json.dumps(_valid_strategy_dict(self._confidence))


class _MarkdownWrappedAdapter(BaseLLMAdapter):
    def generate(self, prompt: str) -> str:
        return "```json\n" + json.dumps(_valid_strategy_dict(0.6)) + "\n```"


class _BrokenThenFixedAdapter(BaseLLMAdapter):
    def __init__(self) -> None:
        self._calls = 0

    def generate(self, prompt: str) -> str:
        self._calls += 1
        if self._calls == 1:
            return "BROKEN"
        return json.dumps(_valid_strategy_dict(0.6))


class _AlwaysBrokenAdapter(BaseLLMAdapter):
    def generate(self, prompt: str) -> str:
        return "not json"


class _ConfidenceExceedingAdapter(BaseLLMAdapter):
    def generate(self, prompt: str) -> str:
        return json.dumps(_valid_strategy_dict(confidence=0.99))


class _BadSignalIdAdapter(BaseLLMAdapter):
    def generate(self, prompt: str) -> str:
        block = _valid_strategy_dict(0.6)
        block["risk_mitigation"][0]["supporting_signals"] = [
            {"signal_id": "SIG-99", "metric_name": "fake", "value": 0.0, "unit": "ratio"},
        ]
        return json.dumps(block)


class _GenericActionAdapter(BaseLLMAdapter):
    def generate(self, prompt: str) -> str:
        block = _valid_strategy_dict(0.6)
        block["short_term_actions"][0]["action"] = "Focus on growth and improve performance across all teams."
        return json.dumps(block)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestStrategyBlockSchema:
    def test_valid_block_parses(self) -> None:
        block = StrategyBlock(**_valid_strategy_dict(0.6))
        assert len(block.short_term_actions) == 3
        assert len(block.mid_term_actions) == 3
        assert "SIG-" in block.long_term_positioning
        assert block.confidence == 0.6

    def test_rejects_extra_fields(self) -> None:
        data = {**_valid_strategy_dict(0.6), "revenue": 100}
        with pytest.raises(Exception):
            StrategyBlock(**data)

    def test_rejects_wrong_short_term_count(self) -> None:
        data = _valid_strategy_dict(0.6)
        data["short_term_actions"] = data["short_term_actions"][:2]
        with pytest.raises(Exception):
            StrategyBlock(**data)

    def test_rejects_wrong_mid_term_count(self) -> None:
        data = _valid_strategy_dict(0.6)
        data["mid_term_actions"] = data["mid_term_actions"][:1]
        with pytest.raises(Exception):
            StrategyBlock(**data)

    def test_rejects_long_term_without_signal_id(self) -> None:
        data = _valid_strategy_dict(0.6)
        data["long_term_positioning"] = "Position as a leader in the healthcare market through sustained investment in product development."
        with pytest.raises(Exception, match="signal ID"):
            StrategyBlock(**data)

    def test_rejects_duplicate_actions(self) -> None:
        data = _valid_strategy_dict(0.6)
        # Copy first short-term action into mid-term
        data["mid_term_actions"][0] = data["short_term_actions"][0].copy()
        with pytest.raises(Exception, match="[Dd]uplicate"):
            StrategyBlock(**data)


class TestStrategyAction:
    def test_rejects_generic_advice(self) -> None:
        with pytest.raises(Exception, match="[Gg]eneric"):
            StrategyAction(
                action="Focus on growth and improve performance across all dimensions of the business.",
                rationale="Growth is important for competitive advantage.",
                supporting_signals=[
                    {"signal_id": "SIG-0", "metric_name": "x", "value": 1.0, "unit": "usd"},
                ],
                priority="high",
            )

    def test_rejects_short_action(self) -> None:
        with pytest.raises(Exception):
            StrategyAction(
                action="Do something",
                rationale="Because we should act on competitor data.",
                supporting_signals=[
                    {"signal_id": "SIG-0", "metric_name": "x", "value": 1.0, "unit": "usd"},
                ],
                priority="high",
            )


# ---------------------------------------------------------------------------
# Validation helper tests
# ---------------------------------------------------------------------------


class TestValidationHelpers:
    def test_signal_ids_in_scope_pass(self) -> None:
        block = StrategyBlock(**_valid_strategy_dict(0.6))
        insight = _make_insight(0.65)
        _validate_signal_ids_in_scope(block, insight)  # no exception

    def test_signal_ids_in_scope_fail(self) -> None:
        data = _valid_strategy_dict(0.6)
        data["risk_mitigation"][0]["supporting_signals"] = [
            {"signal_id": "SIG-99", "metric_name": "fake", "value": 0.0, "unit": "ratio"},
        ]
        block = StrategyBlock(**data)
        insight = _make_insight(0.65)
        with pytest.raises(ValueError, match="SIG-99"):
            _validate_signal_ids_in_scope(block, insight)

    def test_confidence_ceiling_pass(self) -> None:
        block = StrategyBlock(**_valid_strategy_dict(0.6))
        _validate_confidence_ceiling(block, 0.65)

    def test_confidence_ceiling_fail(self) -> None:
        block = StrategyBlock(**_valid_strategy_dict(0.6))
        with pytest.raises(ValueError, match="exceeds"):
            _validate_confidence_ceiling(block, 0.5)


class TestCollectSignalRefs:
    def test_deduplicates(self) -> None:
        insight = _make_insight()
        refs = _collect_signal_refs(insight)
        ids = [r["signal_id"] for r in refs]
        assert len(ids) == len(set(ids))
        assert "SIG-0" in ids
        assert "SIG-1" in ids


# ---------------------------------------------------------------------------
# Service tests
# ---------------------------------------------------------------------------


class TestStrategyGenerator:
    def test_happy_path(self) -> None:
        gen = StrategyGenerator(_MockStrategyAdapter(confidence=0.6))
        insight = _make_insight(confidence=0.65)
        block = gen.generate(insight)

        assert isinstance(block, StrategyBlock)
        assert len(block.short_term_actions) == 3
        assert len(block.mid_term_actions) == 3
        assert block.confidence <= insight.confidence + 1e-6
        assert "SIG-" in block.long_term_positioning

    def test_strips_markdown_fences(self) -> None:
        gen = StrategyGenerator(_MarkdownWrappedAdapter())
        block = gen.generate(_make_insight(0.65))
        assert isinstance(block, StrategyBlock)

    def test_retries_on_bad_json(self) -> None:
        gen = StrategyGenerator(_BrokenThenFixedAdapter(), max_retries=2)
        block = gen.generate(_make_insight(0.65))
        assert isinstance(block, StrategyBlock)

    def test_raises_after_exhausted_retries(self) -> None:
        gen = StrategyGenerator(_AlwaysBrokenAdapter(), max_retries=1)
        with pytest.raises(StrategyGeneratorError) as exc_info:
            gen.generate(_make_insight())
        assert exc_info.value.attempts == 2

    def test_rejects_confidence_exceeding_insight(self) -> None:
        gen = StrategyGenerator(_ConfidenceExceedingAdapter(), max_retries=0)
        with pytest.raises(StrategyGeneratorError, match="Validation error"):
            gen.generate(_make_insight(confidence=0.65))

    def test_rejects_nonexistent_signal_ids(self) -> None:
        gen = StrategyGenerator(_BadSignalIdAdapter(), max_retries=0)
        with pytest.raises(StrategyGeneratorError, match="Validation error"):
            gen.generate(_make_insight())

    def test_rejects_generic_actions(self) -> None:
        gen = StrategyGenerator(_GenericActionAdapter(), max_retries=0)
        with pytest.raises(StrategyGeneratorError, match="Validation error"):
            gen.generate(_make_insight())

    def test_prompt_contains_signal_table(self) -> None:
        gen = StrategyGenerator(_MockStrategyAdapter())
        insight = _make_insight(0.65)
        prompt = gen._build_prompt(insight)
        assert "SIG-0" in prompt
        assert "SIG-1" in prompt
        assert "listed_price_usd_mean" in prompt

    def test_prompt_contains_insight_summary(self) -> None:
        gen = StrategyGenerator(_MockStrategyAdapter())
        insight = _make_insight(0.65)
        prompt = gen._build_prompt(insight)
        assert "INSIGHT BLOCK SUMMARY" in prompt
        assert "opportunity_zones" in prompt
        assert "risk_zones" in prompt
