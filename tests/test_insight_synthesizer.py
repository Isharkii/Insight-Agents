"""Tests for InsightSynthesizer — schema, prompt, validation, fallback."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from app.business_intelligence.context_analyzer import BusinessContext
from app.business_intelligence.insight_synthesizer import (
    InsightBlock,
    InsightSynthesizer,
    InsightSynthesizerError,
    SignalReference,
    _format_signal_table,
    _validate_confidence_ceiling,
    _validate_signal_ids_exist,
)
from app.business_intelligence.intelligence_orchestrator import (
    IntelligenceBundle,
    SignalRecord,
    StageStatus,
)
from llm_synthesis.adapter import BaseLLMAdapter


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


def _make_bundle(
    signals: list[SignalRecord] | None = None,
    confidence: float = 0.65,
    context: BusinessContext | None = _CONTEXT,
) -> IntelligenceBundle:
    return IntelligenceBundle(
        status="partial",
        generated_at=datetime.now(timezone.utc),
        business_context=context,
        signals=signals if signals is not None else _SIGNALS,
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


# A valid InsightBlock JSON that references signals from _SIGNALS
def _valid_insight_block_dict(confidence: float = 0.6) -> dict:
    return {
        "emerging_signals": [
            {
                "title": "Healthcare SaaS pricing benchmark at $49/month",
                "description": "Search extraction identified a market price point of $49/month (SIG-0) suggesting mid-tier positioning for healthcare SaaS products.",
                "supporting_signals": [
                    {"signal_id": "SIG-0", "metric_name": "listed_price_usd_mean", "value": 49.0, "unit": "usd"},
                ],
                "relevance": "high",
            },
            {
                "title": "Growth rate signal at 25% from market sources",
                "description": "Market growth rate of 25% (SIG-1) detected from competitor benchmark sources indicating strong sector momentum.",
                "supporting_signals": [
                    {"signal_id": "SIG-1", "metric_name": "growth_rate_mentioned_pct", "value": 0.25, "unit": "ratio"},
                ],
                "relevance": "high",
            },
        ],
        "macro_summary": (
            "Healthcare AI funding is accelerating (SIG-2) while telemedicine "
            "search interest shows 50K traffic (SIG-3) and clinic SaaS interest "
            "is at 12K (SIG-4).  The macro environment appears supportive for "
            "digital health expansion in emerging markets."
        ),
        "opportunity_zones": [
            {
                "title": "Telemedicine adoption wave in target market",
                "description": "High trend traffic for telemedicine (SIG-3) and clinic SaaS (SIG-4) suggests growing demand in the target healthcare vertical.",
                "supporting_signals": [
                    {"signal_id": "SIG-3", "metric_name": "trend_keyword_traffic", "value": 50000.0, "unit": "count"},
                    {"signal_id": "SIG-4", "metric_name": "trend_keyword_traffic", "value": 12000.0, "unit": "count"},
                ],
            },
        ],
        "risk_zones": [
            {
                "title": "Low confidence in competitor growth signal",
                "description": "The growth rate signal (SIG-1) carries low extraction confidence (0.3) and may not reflect verified competitor data.",
                "supporting_signals": [
                    {"signal_id": "SIG-1", "metric_name": "growth_rate_mentioned_pct", "value": 0.25, "unit": "ratio"},
                ],
            },
        ],
        "momentum_score": 0.6,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Mock adapters
# ---------------------------------------------------------------------------


class _MockSynthesisAdapter(BaseLLMAdapter):
    """Returns a valid InsightBlock JSON."""

    def __init__(self, confidence: float = 0.6) -> None:
        self._confidence = confidence

    def generate(self, prompt: str) -> str:
        return json.dumps(_valid_insight_block_dict(self._confidence))


class _MarkdownWrappedAdapter(BaseLLMAdapter):
    def generate(self, prompt: str) -> str:
        return "```json\n" + json.dumps(_valid_insight_block_dict(0.6)) + "\n```"


class _BrokenThenFixedAdapter(BaseLLMAdapter):
    def __init__(self) -> None:
        self._calls = 0

    def generate(self, prompt: str) -> str:
        self._calls += 1
        if self._calls == 1:
            return "NOT VALID JSON"
        return json.dumps(_valid_insight_block_dict(0.6))


class _AlwaysBrokenAdapter(BaseLLMAdapter):
    def generate(self, prompt: str) -> str:
        return "totally broken output"


class _ConfidenceExceedingAdapter(BaseLLMAdapter):
    """Returns InsightBlock with confidence exceeding bundle ceiling."""

    def generate(self, prompt: str) -> str:
        return json.dumps(_valid_insight_block_dict(confidence=0.99))


class _BadSignalIdAdapter(BaseLLMAdapter):
    """Returns InsightBlock referencing a signal ID that doesn't exist."""

    def generate(self, prompt: str) -> str:
        block = _valid_insight_block_dict(0.6)
        block["risk_zones"][0]["supporting_signals"] = [
            {"signal_id": "SIG-99", "metric_name": "fake", "value": 0.0, "unit": "ratio"},
        ]
        return json.dumps(block)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestInsightBlockSchema:
    def test_valid_block_parses(self) -> None:
        block = InsightBlock(**_valid_insight_block_dict(0.6))
        assert len(block.emerging_signals) == 2
        assert block.momentum_score == 0.6
        assert block.confidence == 0.6
        assert "SIG-" in block.macro_summary

    def test_rejects_extra_fields(self) -> None:
        data = {**_valid_insight_block_dict(0.6), "revenue": 1_000_000}
        with pytest.raises(Exception):
            InsightBlock(**data)

    def test_rejects_empty_emerging_signals(self) -> None:
        data = {**_valid_insight_block_dict(0.6), "emerging_signals": []}
        with pytest.raises(Exception):
            InsightBlock(**data)

    def test_rejects_macro_summary_without_signal_id(self) -> None:
        data = _valid_insight_block_dict(0.6)
        data["macro_summary"] = "The macro environment is broadly positive with no specific references."
        with pytest.raises(Exception, match="signal ID"):
            InsightBlock(**data)

    def test_rejects_bad_signal_id_format(self) -> None:
        data = _valid_insight_block_dict(0.6)
        data["emerging_signals"][0]["supporting_signals"] = [
            {"signal_id": "BAD", "metric_name": "x", "value": 1.0, "unit": "usd"},
        ]
        with pytest.raises(Exception):
            InsightBlock(**data)

    def test_momentum_score_bounds(self) -> None:
        data = _valid_insight_block_dict(0.6)
        data["momentum_score"] = 1.5
        with pytest.raises(Exception):
            InsightBlock(**data)

    def test_confidence_bounds(self) -> None:
        data = _valid_insight_block_dict(0.6)
        data["confidence"] = -0.1
        with pytest.raises(Exception):
            InsightBlock(**data)


class TestSignalReference:
    def test_valid_reference(self) -> None:
        ref = SignalReference(signal_id="SIG-0", metric_name="price", value=49.0, unit="usd")
        assert ref.signal_id == "SIG-0"

    def test_rejects_short_signal_id(self) -> None:
        with pytest.raises(Exception):
            SignalReference(signal_id="S-0", metric_name="price", value=49.0, unit="usd")


# ---------------------------------------------------------------------------
# Validation helper tests
# ---------------------------------------------------------------------------


class TestValidationHelpers:
    def test_signal_ids_exist_pass(self) -> None:
        block = InsightBlock(**_valid_insight_block_dict(0.6))
        _validate_signal_ids_exist(block, _SIGNALS)  # no exception

    def test_signal_ids_exist_fail(self) -> None:
        block = InsightBlock(**_valid_insight_block_dict(0.6))
        with pytest.raises(ValueError, match="SIG-4"):
            # Only 3 signals → SIG-3 and SIG-4 are invalid
            _validate_signal_ids_exist(block, _SIGNALS[:3])

    def test_confidence_ceiling_pass(self) -> None:
        block = InsightBlock(**_valid_insight_block_dict(0.6))
        _validate_confidence_ceiling(block, 0.65)  # no exception

    def test_confidence_ceiling_fail(self) -> None:
        block = InsightBlock(**_valid_insight_block_dict(0.6))
        with pytest.raises(ValueError, match="exceeds"):
            _validate_confidence_ceiling(block, 0.5)


class TestSignalTable:
    def test_format_includes_ids(self) -> None:
        table = _format_signal_table(_SIGNALS)
        assert "SIG-0" in table
        assert "SIG-4" in table
        assert "listed_price_usd_mean" in table


# ---------------------------------------------------------------------------
# Service tests
# ---------------------------------------------------------------------------


class TestInsightSynthesizer:
    def test_happy_path(self) -> None:
        synth = InsightSynthesizer(_MockSynthesisAdapter(confidence=0.6))
        bundle = _make_bundle(confidence=0.65)
        block = synth.synthesize(bundle)

        assert isinstance(block, InsightBlock)
        assert block.confidence <= bundle.confidence + 1e-6
        assert len(block.emerging_signals) >= 1
        assert len(block.opportunity_zones) >= 1
        assert len(block.risk_zones) >= 1
        assert "SIG-" in block.macro_summary

    def test_strips_markdown_fences(self) -> None:
        synth = InsightSynthesizer(_MarkdownWrappedAdapter())
        block = synth.synthesize(_make_bundle(confidence=0.65))
        assert isinstance(block, InsightBlock)

    def test_retries_on_bad_json(self) -> None:
        synth = InsightSynthesizer(_BrokenThenFixedAdapter(), max_retries=2)
        block = synth.synthesize(_make_bundle(confidence=0.65))
        assert isinstance(block, InsightBlock)

    def test_raises_after_exhausted_retries(self) -> None:
        synth = InsightSynthesizer(_AlwaysBrokenAdapter(), max_retries=1)
        with pytest.raises(InsightSynthesizerError) as exc_info:
            synth.synthesize(_make_bundle())
        assert exc_info.value.attempts == 2

    def test_rejects_confidence_exceeding_bundle(self) -> None:
        synth = InsightSynthesizer(
            _ConfidenceExceedingAdapter(), max_retries=0,
        )
        with pytest.raises(InsightSynthesizerError, match="Validation error"):
            synth.synthesize(_make_bundle(confidence=0.65))

    def test_rejects_nonexistent_signal_ids(self) -> None:
        synth = InsightSynthesizer(_BadSignalIdAdapter(), max_retries=0)
        with pytest.raises(InsightSynthesizerError, match="Validation error"):
            synth.synthesize(_make_bundle())

    def test_rejects_empty_signals(self) -> None:
        synth = InsightSynthesizer(_MockSynthesisAdapter())
        with pytest.raises(ValueError, match="no signals"):
            synth.synthesize(_make_bundle(signals=[]))

    def test_prompt_contains_signal_table(self) -> None:
        synth = InsightSynthesizer(_MockSynthesisAdapter(confidence=0.6))
        bundle = _make_bundle(confidence=0.65)
        prompt = synth._build_prompt(bundle)
        assert "SIG-0" in prompt
        assert "SIG-4" in prompt
        assert "listed_price_usd_mean" in prompt
        assert "BUNDLE CONFIDENCE" in prompt or "bundle_confidence" in prompt

    def test_prompt_contains_business_context(self) -> None:
        synth = InsightSynthesizer(_MockSynthesisAdapter(confidence=0.6))
        bundle = _make_bundle(confidence=0.65)
        prompt = synth._build_prompt(bundle)
        assert "healthcare technology" in prompt
        assert "BUSINESS CONTEXT" in prompt

    def test_prompt_without_context(self) -> None:
        synth = InsightSynthesizer(_MockSynthesisAdapter(confidence=0.6))
        bundle = _make_bundle(confidence=0.65, context=None)
        prompt = synth._build_prompt(bundle)
        assert "BUSINESS CONTEXT" not in prompt
        assert "SIG-0" in prompt  # signals still present
