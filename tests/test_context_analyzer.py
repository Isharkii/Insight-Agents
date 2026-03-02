"""Tests for the BusinessContext schema and ContextAnalyzer service."""

from __future__ import annotations

import json

import pytest

from app.business_intelligence.context_analyzer import (
    BusinessContext,
    ContextAnalyzer,
    ContextAnalyzerError,
)
from llm_synthesis.adapter import BaseLLMAdapter


# ---------------------------------------------------------------------------
# Fixtures — deterministic mock adapters
# ---------------------------------------------------------------------------

_VALID_CONTEXT = {
    "industry": "healthcare technology",
    "business_model": "saas",
    "target_market": "SMB healthcare clinics in India",
    "macro_dependencies": [
        "Indian healthcare regulation (NMC / ABDM)",
        "USD/INR exchange rate",
        "Digital health adoption subsidies",
    ],
    "search_intents": [
        "AI SaaS healthcare clinic management India market size 2025",
        "top competitors clinic management software India",
        "ABDM digital health compliance requirements SaaS",
        "healthcare SaaS churn benchmarks emerging markets",
        "India clinic EHR adoption rate trends",
    ],
    "risk_factors": [
        "Regulatory dependency on ABDM integration mandates",
        "Single-market concentration in India",
        "Low willingness-to-pay among small clinics",
    ],
}


class _MockContextAdapter(BaseLLMAdapter):
    """Returns a valid BusinessContext JSON response."""

    def generate(self, prompt: str) -> str:
        return json.dumps(_VALID_CONTEXT)


class _MarkdownWrappedAdapter(BaseLLMAdapter):
    """Returns valid JSON wrapped in markdown code fences."""

    def generate(self, prompt: str) -> str:
        return "```json\n" + json.dumps(_VALID_CONTEXT) + "\n```"


class _BrokenThenFixedAdapter(BaseLLMAdapter):
    """Fails on the first call, succeeds on the second."""

    def __init__(self) -> None:
        self._calls = 0

    def generate(self, prompt: str) -> str:
        self._calls += 1
        if self._calls == 1:
            return "NOT JSON AT ALL"
        return json.dumps(_VALID_CONTEXT)


class _AlwaysBrokenAdapter(BaseLLMAdapter):
    """Always returns invalid output."""

    def generate(self, prompt: str) -> str:
        return "totally broken response"


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestBusinessContextSchema:
    def test_valid_context_parses(self) -> None:
        ctx = BusinessContext(**_VALID_CONTEXT)
        assert ctx.industry == "healthcare technology"
        assert ctx.business_model == "saas"
        assert len(ctx.search_intents) == 5
        assert len(ctx.macro_dependencies) >= 1
        assert len(ctx.risk_factors) >= 1

    def test_business_model_normalised(self) -> None:
        data = {**_VALID_CONTEXT, "business_model": "SaaS"}
        ctx = BusinessContext(**data)
        assert ctx.business_model == "saas"

    def test_business_model_with_spaces_normalised(self) -> None:
        data = {**_VALID_CONTEXT, "business_model": "E Commerce"}
        ctx = BusinessContext(**data)
        assert ctx.business_model == "e_commerce"

    def test_rejects_extra_fields(self) -> None:
        data = {**_VALID_CONTEXT, "revenue": 1_000_000}
        with pytest.raises(Exception):  # Pydantic ValidationError
            BusinessContext(**data)

    def test_rejects_wrong_search_intent_count(self) -> None:
        data = {**_VALID_CONTEXT, "search_intents": ["only one"]}
        with pytest.raises(Exception):
            BusinessContext(**data)

    def test_rejects_empty_industry(self) -> None:
        data = {**_VALID_CONTEXT, "industry": ""}
        with pytest.raises(Exception):
            BusinessContext(**data)


# ---------------------------------------------------------------------------
# Service tests
# ---------------------------------------------------------------------------


class TestContextAnalyzer:
    def test_happy_path(self) -> None:
        analyzer = ContextAnalyzer(_MockContextAdapter())
        result = analyzer.analyze("AI SaaS for healthcare clinics in India")

        assert isinstance(result, BusinessContext)
        assert result.business_model == "saas"
        assert result.target_market == "SMB healthcare clinics in India"
        assert len(result.search_intents) == 5

    def test_strips_markdown_fences(self) -> None:
        analyzer = ContextAnalyzer(_MarkdownWrappedAdapter())
        result = analyzer.analyze("AI SaaS for healthcare clinics in India")
        assert result.business_model == "saas"

    def test_retries_on_bad_json(self) -> None:
        analyzer = ContextAnalyzer(_BrokenThenFixedAdapter(), max_retries=2)
        result = analyzer.analyze("AI SaaS for healthcare clinics in India")
        assert result.business_model == "saas"

    def test_raises_after_exhausted_retries(self) -> None:
        analyzer = ContextAnalyzer(_AlwaysBrokenAdapter(), max_retries=1)
        with pytest.raises(ContextAnalyzerError) as exc_info:
            analyzer.analyze("anything")
        assert exc_info.value.attempts == 2

    def test_rejects_empty_description(self) -> None:
        analyzer = ContextAnalyzer(_MockContextAdapter())
        with pytest.raises(ValueError, match="must not be empty"):
            analyzer.analyze("   ")


# ---------------------------------------------------------------------------
# Example usage (runnable as a script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Swap _MockContextAdapter for OpenAILLMAdapter in production:
    #
    #   from llm_synthesis.adapter import OpenAILLMAdapter
    #   adapter = OpenAILLMAdapter(model="gpt-4o")
    #
    adapter = _MockContextAdapter()
    analyzer = ContextAnalyzer(adapter)

    ctx = analyzer.analyze("AI SaaS for healthcare clinics in India")

    print("=== Extracted Business Context ===")
    print(f"Industry:       {ctx.industry}")
    print(f"Business Model: {ctx.business_model}")
    print(f"Target Market:  {ctx.target_market}")
    print(f"Macro Deps:     {ctx.macro_dependencies}")
    print(f"Risk Factors:   {ctx.risk_factors}")
    print(f"Search Intents:")
    for i, q in enumerate(ctx.search_intents, 1):
        print(f"  {i}. {q}")
