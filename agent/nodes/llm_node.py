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

from agent.graph_config import (
    KPI_KEY_BY_BUSINESS_TYPE,
    graph_node_config_for_business_type,
    signal_name_for_state_key,
)
from agent.nodes.node_result import confidence_of, payload_of, status_of, warnings_of
from agent.state import AgentState
from db.config import load_env_files
from llm_synthesis.adapter import BaseLLMAdapter, MockLLMAdapter, OpenAILLMAdapter
from llm_synthesis.prompt_builder import SynthesisPromptBuilder
from llm_synthesis.retry import generate_with_retry
from llm_synthesis.schema import (
    ConfidenceAdjustment,
    EnvelopeDiagnostics,
    InsightOutput as FinalInsightResponse,
)

_prompt_builder = SynthesisPromptBuilder()


def _resolve_kpi_result(state: AgentState) -> Any:
    business_type = str(state.get("business_type") or "").lower()
    preferred_key = KPI_KEY_BY_BUSINESS_TYPE.get(business_type)
    if preferred_key:
        return state.get(preferred_key)

    for key in ("kpi_data", "saas_kpi_data", "ecommerce_kpi_data", "agency_kpi_data"):
        value = state.get(key)
        if value is not None:
            return value
    return None


def _usable_payload(value: Any) -> dict[str, Any]:
    """Extract payload only from success envelopes."""
    return payload_of(value) if status_of(value) == "success" else {}


def _dedupe_text(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for raw in items:
        text = str(raw).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _collect_diagnostics(state: AgentState) -> dict[str, Any]:
    """Collect warnings, missing signals, and confidence adjustments."""
    all_warnings: list[str] = []
    missing_signal: list[str] = []
    adjustments: list[dict[str, Any]] = []

    config = graph_node_config_for_business_type(str(state.get("business_type") or ""))
    required_keys = config.required
    optional_keys = config.optional
    ordered_keys: list[str] = list(dict.fromkeys((*required_keys, *optional_keys)))

    for key in ordered_keys:
        is_required = key in required_keys
        value = state.get(key)
        signal = signal_name_for_state_key(key)

        if value is None:
            if is_required:
                missing_signal.append(signal)
                all_warnings.append(f"Required signal '{signal}' is unavailable.")
                adjustments.append(
                    {
                        "signal": signal,
                        "delta": -0.35,
                        "reason": "required_signal_unavailable",
                    }
                )
            continue

        status = status_of(value)
        node_warnings = warnings_of(value)
        if node_warnings:
            all_warnings.extend(node_warnings)

        if status in {"skipped", "failed"}:
            missing_signal.append(signal)
            penalty = -0.35 if is_required else -0.10
            adjustments.append(
                {
                    "signal": signal,
                    "delta": penalty,
                    "reason": (
                        "required_signal_unavailable"
                        if is_required
                        else "optional_signal_unavailable"
                    ),
                }
            )
            all_warnings.append(
                (
                    f"Required signal '{signal}' is {status}."
                    if is_required
                    else f"Optional signal '{signal}' is {status}; partial coverage applied."
                )
            )
            continue

        confidence = confidence_of(value)
        if 0.0 < confidence < 1.0:
            delta = round(confidence - 1.0, 6)
            adjustments.append(
                {
                    "signal": signal,
                    "delta": delta,
                    "reason": "upstream_confidence_penalty",
                }
            )

    # Dataset-level confidence penalty propagates into the final score.
    dataset_confidence_raw = state.get("dataset_confidence")
    try:
        dataset_confidence = float(dataset_confidence_raw)
    except (TypeError, ValueError):
        dataset_confidence = 1.0
    dataset_confidence = max(0.0, min(1.0, dataset_confidence))
    if dataset_confidence < 1.0:
        adjustments.append(
            {
                "signal": "dataset",
                "delta": round(dataset_confidence - 1.0, 6),
                "reason": "dataset_confidence_penalty",
            }
        )

    # Surface ingestion warnings in diagnostics.
    ingestion_warnings = state.get("ingestion_warnings")
    if isinstance(ingestion_warnings, list):
        all_warnings.extend(str(item) for item in ingestion_warnings if str(item).strip())

    confidence_delta_total = sum(float(item["delta"]) for item in adjustments)
    confidence_score = max(0.0, min(1.0, round(1.0 + confidence_delta_total, 6)))

    return {
        "warnings": _dedupe_text(all_warnings),
        "missing_signal": _dedupe_text(missing_signal),
        "confidence_adjustments": adjustments,
        "confidence_score": confidence_score,
    }


def _derive_pipeline_status(state: AgentState) -> str:
    existing = str(state.get("pipeline_status") or "").strip().lower()
    if existing in {"success", "partial", "failed"}:
        return existing

    config = graph_node_config_for_business_type(str(state.get("business_type") or ""))
    required_keys = config.required
    optional_keys = config.optional

    for key in required_keys:
        if status_of(state.get(key)) != "success":
            return "failed"

    for key in optional_keys:
        value = state.get(key)
        if value is None:
            continue
        if status_of(value) != "success":
            return "partial"

    return "success"


def _build_adapter() -> BaseLLMAdapter:
    """Instantiate the adapter selected by the LLM_ADAPTER env var."""
    adapter_name = os.getenv("LLM_ADAPTER", "openai").strip().lower()
    if adapter_name == "mock":
        return MockLLMAdapter()

    model_env = os.getenv("LLM_MODEL", "").strip()
    model_name = model_env or "gpt-4o-mini"

    return OpenAILLMAdapter(
        model=model_name,
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

    kpi_data = _usable_payload(_resolve_kpi_result(state))
    forecast_data = _usable_payload(state.get("forecast_data"))
    risk_data = _usable_payload(state.get("risk_data"))
    root_cause = _usable_payload(state.get("root_cause"))
    pipeline_status = _derive_pipeline_status(state)
    diagnostics = _collect_diagnostics(state)

    prompt = _prompt_builder.build_prompt(
        kpi_data=kpi_data,
        forecast_data=forecast_data,
        risk_data=risk_data,
        root_cause=root_cause,
        segmentation=_usable_payload(state.get("segmentation")),
        prioritization=state.get("prioritization") or {},
        confidence_score=float(diagnostics.get("confidence_score", 1.0)),
        missing_signals=diagnostics.get("missing_signal", []),
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

    diagnostics_model = EnvelopeDiagnostics(
        warnings=[str(item) for item in diagnostics.get("warnings", [])],
        confidence_score=float(diagnostics.get("confidence_score", 1.0)),
        missing_signal=[str(item) for item in diagnostics.get("missing_signal", [])],
        confidence_adjustments=[
            ConfidenceAdjustment(
                signal=str(item.get("signal") or "unknown"),
                delta=float(item.get("delta") or 0.0),
                reason=str(item.get("reason") or "unspecified"),
            )
            for item in diagnostics.get("confidence_adjustments", [])
            if isinstance(item, dict)
        ],
    )

    # Enforce: deterministic confidence always overrides LLM self-assessment.
    # The LLM must not inflate confidence beyond what the signals support.
    deterministic_confidence = float(diagnostics.get("confidence_score", 1.0))
    enforced_confidence = min(
        final_payload.confidence_score,
        deterministic_confidence,
    )

    try:
        final_payload = final_payload.model_copy(
            update={
                "confidence_score": enforced_confidence,
                "pipeline_status": pipeline_status,
                "diagnostics": diagnostics_model,
            }
        )
        final_response = final_payload.model_dump_json()
        _validate_final_response_contract(final_response)
    except (json.JSONDecodeError, TypeError, ValidationError, ValueError) as exc:
        fallback = FinalInsightResponse.failure(
            reason=str(exc),
            pipeline_status=pipeline_status,
        ).model_copy(update={"diagnostics": diagnostics_model})
        final_response = fallback.model_dump_json()

    return {
        **state,
        "pipeline_status": pipeline_status,
        "final_response": final_response,
        "envelope_diagnostics": diagnostics,
    }
