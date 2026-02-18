"""LLM node for final structured insight synthesis.

Builds the synthesis prompt, calls the LLM through retry+validation,
and writes the final serialized response to ``state[\"final_response\"]``.
"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import ValidationError

from agent.state import AgentState
from db.config import load_env_files
from llm_synthesis.adapter import BaseLLMAdapter, MockLLMAdapter, OpenAILLMAdapter
from llm_synthesis.prompt_builder import SynthesisPromptBuilder
from llm_synthesis.retry import generate_with_retry
from llm_synthesis.schema import InsightOutput
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

    LLM_ADAPTER=mock   -> MockLLMAdapter  (testing, no API key required)
    LLM_ADAPTER=openai -> OpenAILLMAdapter (default)
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


def _validate_final_response_contract(final_response: str) -> None:
    """Ensure serialized output conforms to the InsightOutput contract."""
    try:
        payload = json.loads(final_response)
        InsightOutput.model_validate(payload)
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        raise ValueError(
            f"Final response must match InsightOutput schema: {exc}"
        ) from exc


def llm_node(state: AgentState) -> AgentState:
    """LangGraph node: synthesize upstream outputs into a structured insight.

    Flow:
        1. Build prompt via SynthesisPromptBuilder.
        2. Call generate_with_retry() -> validated InsightOutput.
        3. Serialize to JSON and validate final response contract.
        4. Store in ``state[\"final_response\"]``.

    Writes:
        state["final_response"] (str): InsightOutput JSON on success,
            error message string on generation failure.

    Raises:
        ValueError: If the final serialized output does not match InsightOutput.
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
        _validate_final_response_contract(final_response)
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        final_response = f"LLM generation failed: {exc}"

    return {**state, "final_response": final_response}
