"""LLM node for final structured insight synthesis.

Builds the synthesis prompt, calls the LLM through retry+validation,
and writes the final serialized response to ``state[\"final_response\"]``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from agent.state import AgentState
from db.config import load_env_files
from llm_synthesis.adapter import BaseLLMAdapter, MockLLMAdapter, OpenAILLMAdapter
from llm_synthesis.prompt_builder import SynthesisPromptBuilder
from llm_synthesis.retry import generate_with_retry
from llm_synthesis.schema import InsightOutput as FinalInsightResponse
from llm_synthesis.validator import validate_llm_output  # noqa: F401  (used via generate_with_retry)

_prompt_builder = SynthesisPromptBuilder()
logger = logging.getLogger(__name__)
_CONTEXT_STRENGTH_THRESHOLD = 3


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
    """Ensure serialized output conforms to the FinalInsightResponse contract."""
    try:
        payload = json.loads(final_response)
        FinalInsightResponse.model_validate(payload)
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        raise ValueError(
            f"Final response must match FinalInsightResponse schema: {exc}"
        ) from exc


def _validate_signal_context_or_raise(state: AgentState) -> tuple[dict, dict, dict, dict]:
    """Validate required upstream signal context before LLM invocation."""
    kpi_data = _resolve_kpi_data(state)
    forecast_data = state.get("forecast_data")
    risk_data = state.get("risk_data")
    root_cause = state.get("root_cause")
    business_type = str(state.get("business_type") or "")
    missing_fields: list[str] = []

    has_root_cause_insights = False
    if isinstance(root_cause, dict):
        insights = root_cause.get("insights")
        root_causes = root_cause.get("root_causes")
        has_root_cause_insights = bool(insights) or bool(root_causes)

    if not isinstance(kpi_data, dict) or not kpi_data:
        missing_fields.append("kpi_data")
    if not isinstance(forecast_data, dict) or not forecast_data:
        missing_fields.append("forecast_data")
    if not isinstance(risk_data, dict) or risk_data.get("risk_score") is None:
        missing_fields.append("risk_data.risk_score")
    if not isinstance(root_cause, dict) or not has_root_cause_insights:
        missing_fields.append("root_cause.insights")

    if missing_fields:
        logger.error(
            "Pipeline critical failure at llm_node: incomplete signal context",
            extra={
                "stage_name": "llm_node",
                "missing_fields": missing_fields,
                "business_type": business_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise ValueError("LLM node received incomplete signal context.")

    return kpi_data, forecast_data, risk_data, root_cause


def _count_non_zero_signals(payload: Any) -> int:
    if isinstance(payload, bool):
        return 0
    if isinstance(payload, (int, float)):
        return int(payload != 0)
    if isinstance(payload, dict):
        return sum(_count_non_zero_signals(value) for value in payload.values())
    if isinstance(payload, list):
        return sum(_count_non_zero_signals(value) for value in payload)
    return 0


def _context_strength(
    *,
    kpi_data: dict,
    forecast_data: dict,
    risk_data: dict,
    root_cause: dict,
) -> tuple[int, int, bool, bool]:
    non_zero_signals = (
        _count_non_zero_signals(kpi_data)
        + _count_non_zero_signals(forecast_data)
    )
    has_risk_score = risk_data.get("risk_score") is not None
    has_root_cause_factors = bool(
        root_cause.get("root_causes") or root_cause.get("contributing_factors")
    )
    strength = non_zero_signals + int(has_risk_score) + int(has_root_cause_factors)
    return strength, non_zero_signals, has_risk_score, has_root_cause_factors


def _validate_context_strength_or_raise(
    *,
    kpi_data: dict,
    forecast_data: dict,
    risk_data: dict,
    root_cause: dict,
    business_type: str,
) -> None:
    strength, non_zero_signals, has_risk_score, has_root_cause_factors = _context_strength(
        kpi_data=kpi_data,
        forecast_data=forecast_data,
        risk_data=risk_data,
        root_cause=root_cause,
    )
    if strength < _CONTEXT_STRENGTH_THRESHOLD:
        missing_fields: list[str] = []
        if non_zero_signals < 1:
            missing_fields.append("non_zero_signals")
        if not has_risk_score:
            missing_fields.append("risk_score")
        if not has_root_cause_factors:
            missing_fields.append("root_cause_factors")
        logger.error(
            "Pipeline critical failure at llm_node: insufficient analytical signal",
            extra={
                "stage_name": "llm_node",
                "missing_fields": missing_fields,
                "business_type": business_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise ValueError("Insufficient analytical signal for LLM synthesis.")


def llm_node(state: AgentState) -> AgentState:
    """LangGraph node: synthesize upstream outputs into a structured insight.

    Flow:
        1. Build prompt via SynthesisPromptBuilder.
        2. Call generate_with_retry() and wrap as FinalInsightResponse.
        3. Serialize to JSON and validate final response contract.
        4. Store in ``state[\"final_response\"]``.

    Writes:
        state["final_response"] (str): FinalInsightResponse JSON.

    Raises:
        ValueError: If the final serialized output does not match FinalInsightResponse.
    """
    load_env_files()
    business_type = str(state.get("business_type") or "")
    kpi_data, forecast_data, risk_data, root_cause = _validate_signal_context_or_raise(
        state
    )
    _validate_context_strength_or_raise(
        kpi_data=kpi_data,
        forecast_data=forecast_data,
        risk_data=risk_data,
        root_cause=root_cause,
        business_type=business_type,
    )

    prompt = _prompt_builder.build_prompt(
        kpi_data=kpi_data,
        forecast_data=forecast_data,
        risk_data=risk_data,
        root_cause=root_cause,
        segmentation=state.get("segmentation") or {},
        prioritization=state.get("prioritization") or {},
    )

    adapter = _build_adapter()

    try:
        synthesis = generate_with_retry(adapter, prompt)
        final_payload = FinalInsightResponse.model_validate(synthesis.model_dump())
    except Exception as error:  # noqa: BLE001
        final_payload = FinalInsightResponse.failure(reason=str(error))

    if not isinstance(final_payload.confidence_score, float):
        raise ValueError("FinalInsightResponse.confidence_score must be numeric.")

    final_response = final_payload.model_dump_json()
    _validate_final_response_contract(final_response)

    return {**state, "final_response": final_response}
