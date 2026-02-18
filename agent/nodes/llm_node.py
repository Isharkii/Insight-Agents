"""
agent/nodes/llm_node.py

LLM Node: builds a structured prompt via SynthesisPromptBuilder, calls the
LLM through generate_with_retry(), and writes the validated SynthesisOutput
JSON to state["final_response"].

Prompt construction  → llm_synthesis.prompt_builder.SynthesisPromptBuilder
Adapter selection    → llm_synthesis.adapter.OpenAILLMAdapter / MockLLMAdapter
Generation + retry   → llm_synthesis.retry.generate_with_retry
Output validation    → llm_synthesis.validator.validate_llm_output
                       (called internally by generate_with_retry)
"""

from __future__ import annotations

import os
from typing import Any

from agent.state import AgentState
from db.config import load_env_files
from llm_synthesis.adapter import BaseLLMAdapter, MockLLMAdapter, OpenAILLMAdapter
from llm_synthesis.prompt_builder import SynthesisPromptBuilder
from llm_synthesis.retry import generate_with_retry
from llm_synthesis.validator import validate_llm_output  # noqa: F401  (used via generate_with_retry)

_prompt_builder = SynthesisPromptBuilder()


def _resolve_kpi_data(state: AgentState) -> Any:
    """Return the first populated KPI namespace from state."""
    for key in ("saas_kpi_data", "ecommerce_kpi_data", "agency_kpi_data"):
        value = state.get(key)
        if value is not None:
            return value
    return {}


def _build_adapter() -> BaseLLMAdapter:
    """Instantiate the adapter selected by the LLM_ADAPTER env var.

    LLM_ADAPTER=mock   → MockLLMAdapter  (testing, no API key required)
    LLM_ADAPTER=openai → OpenAILLMAdapter (default)
    """
    adapter_name = os.getenv("LLM_ADAPTER", "openai").strip().lower()
    if adapter_name == "mock":
        return MockLLMAdapter()

    return OpenAILLMAdapter(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL") or None,
    )


def llm_node(state: AgentState) -> AgentState:
    """LangGraph node: synthesise upstream outputs into a structured insight.

    Flow:
        1. Build prompt via SynthesisPromptBuilder (single prompt-building path).
        2. Call generate_with_retry() → validated SynthesisOutput.
        3. Serialise to JSON and store in state["final_response"].

    Writes:
        state["final_response"] (str) — SynthesisOutput JSON on success,
                                        error message string on failure.
    """
    load_env_files()

    prompt = _prompt_builder.build_prompt(
        kpi_data=_resolve_kpi_data(state),
        forecast_data=state.get("forecast_data") or {},
        risk_data=state.get("risk_data") or {},
        root_cause=state.get("root_cause") or {},
        segmentation=state.get("segmentation") or {},
        prioritization=state.get("prioritization") or {},
    )

    adapter = _build_adapter()

    try:
        synthesis = generate_with_retry(adapter, prompt)
        final_response = synthesis.model_dump_json()
    except Exception as exc:  # noqa: BLE001
        final_response = f"LLM generation failed: {exc}"

    return {**state, "final_response": final_response}
