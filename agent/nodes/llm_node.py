"""LLM node for final structured insight synthesis.

Builds the synthesis prompt from available upstream signals only, calls the
LLM through retry+validation, and writes the serialized response to
``state[\"final_response\"]``.
"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import ValidationError

from agent.nodes.node_result import payload_of, status_of
from agent.state import AgentState
from db.config import load_env_files
from llm_synthesis.adapter import BaseLLMAdapter, MockLLMAdapter, OpenAILLMAdapter
from llm_synthesis.prompt_builder import SynthesisPromptBuilder
from llm_synthesis.retry import generate_with_retry
from llm_synthesis.schema import InsightOutput as FinalInsightResponse

_prompt_builder = SynthesisPromptBuilder()

_KPI_KEY_BY_BUSINESS_TYPE: dict[str, str] = {
    "saas": "saas_kpi_data",
    "ecommerce": "ecommerce_kpi_data",
    "agency": "agency_kpi_data",
}


def _resolve_kpi_result(state: AgentState) -> Any:
    business_type = str(state.get("business_type") or "").lower()
    preferred_key = _KPI_KEY_BY_BUSINESS_TYPE.get(business_type)
    if preferred_key:
        return state.get(preferred_key)

    for key in ("saas_kpi_data", "ecommerce_kpi_data", "agency_kpi_data"):
        value = state.get(key)
        if value is not None:
            return value
    return None


def _success_payload(value: Any) -> dict[str, Any]:
    return payload_of(value) if status_of(value) == "success" else {}


def _derive_pipeline_status(state: AgentState) -> str:
    statuses = [
        status_of(_resolve_kpi_result(state)),
        status_of(state.get("forecast_data")),
        status_of(state.get("risk_data")),
        status_of(state.get("root_cause")),
    ]
    success_count = sum(status == "success" for status in statuses)
    if success_count == len(statuses):
        return "success"
    if success_count > 0:
        return "partial"
    return "failed"


def _build_adapter() -> BaseLLMAdapter:
    """Instantiate the adapter selected by the LLM_ADAPTER env var."""
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
    payload = json.loads(final_response)
    FinalInsightResponse.model_validate(payload)


def llm_node(state: AgentState) -> AgentState:
    """LangGraph node: synthesize available outputs into a structured insight."""
    load_env_files()

    kpi_data = _success_payload(_resolve_kpi_result(state))
    forecast_data = _success_payload(state.get("forecast_data"))
    risk_data = _success_payload(state.get("risk_data"))
    root_cause = _success_payload(state.get("root_cause"))
    pipeline_status = _derive_pipeline_status(state)

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
        final_payload = FinalInsightResponse.failure(
            reason=str(error),
            pipeline_status=pipeline_status,
        )

    try:
        final_payload = final_payload.model_copy(update={"pipeline_status": pipeline_status})
        final_response = final_payload.model_dump_json()
        _validate_final_response_contract(final_response)
    except (json.JSONDecodeError, TypeError, ValidationError, ValueError) as exc:
        fallback = FinalInsightResponse.failure(
            reason=str(exc),
            pipeline_status=pipeline_status,
        )
        final_response = fallback.model_dump_json()

    return {
        **state,
        "pipeline_status": pipeline_status,
        "final_response": final_response,
    }
