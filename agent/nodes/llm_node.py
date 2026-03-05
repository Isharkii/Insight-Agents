"""LLM node for final structured insight synthesis.

Builds the synthesis prompt from available upstream signals only, calls the
LLM through retry+validation, and writes the serialized response to
``state[\"final_response\"]``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from pydantic import ValidationError

from agent.graph_config import (
    KPI_KEY_BY_BUSINESS_TYPE,
    graph_node_config_for_business_type,
    signal_name_for_state_key,
)
from agent.nodes.node_result import payload_of, status_of, warnings_of
from agent.signal_integrity import UnifiedSignalIntegrity
from agent.state import AgentState
from db.config import load_env_files
from llm_synthesis.adapter import BaseLLMAdapter, MockLLMAdapter, OpenAILLMAdapter
from llm_synthesis.prompt_builder import SynthesisPromptBuilder
from llm_synthesis.retry import LLMRetryExhaustedError, generate_with_retry
from llm_synthesis.schema import (
    InsightOutput as FinalInsightResponse,
    set_self_analysis_mode,
)
from llm_synthesis.validator import LLMOutputValidationError

_prompt_builder = SynthesisPromptBuilder()
logger = logging.getLogger(__name__)


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
            continue

        status = status_of(value)
        node_warnings = warnings_of(value)
        if node_warnings:
            all_warnings.extend(node_warnings)

        if status in {"skipped", "failed"}:
            missing_signal.append(signal)
            all_warnings.append(
                (
                    f"Required signal '{signal}' is {status}."
                    if is_required
                    else f"Optional signal '{signal}' is {status}; partial coverage applied."
                )
            )

    # Surface ingestion warnings in diagnostics.
    ingestion_warnings = state.get("ingestion_warnings")
    if isinstance(ingestion_warnings, list):
        all_warnings.extend(str(item) for item in ingestion_warnings if str(item).strip())

    integrity = UnifiedSignalIntegrity.compute(state)
    integrity_scores = UnifiedSignalIntegrity.score_vector_from_integrity(integrity)
    confidence_score = float(integrity.get("overall_score") or 0.0)
    integrity_adjustments = integrity.get("confidence_adjustments")
    if isinstance(integrity_adjustments, list):
        adjustments.extend(
            item
            for item in integrity_adjustments
            if isinstance(item, dict)
        )
    if not bool(integrity.get("kpi_gate_passed", True)):
        all_warnings.append(
            "KPI integrity gate failed (kpi_score < 0.3); confidence forced to 0."
        )

    return {
        "warnings": _dedupe_text(all_warnings),
        "missing_signal": _dedupe_text(missing_signal),
        "confidence_adjustments": adjustments,
        "confidence_score": max(0.0, min(1.0, round(confidence_score, 6))),
        "signal_integrity_scores": integrity_scores,
        "signal_integrity": integrity,
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


def _public_failure_reason(error: Exception) -> str:
    """Map internal synthesis failures to concise, user-facing reasons."""
    if isinstance(error, LLMRetryExhaustedError):
        stage = str(getattr(error.last_error, "stage", "")).strip().lower()
        if stage == "schema":
            return (
                "LLM output did not satisfy the required "
                "analysis schema after retries."
            )
        if stage == "json_parse":
            return (
                "LLM output returned invalid JSON format "
                "after retries."
            )
        return "LLM output failed validation after retries."

    if isinstance(error, LLMOutputValidationError):
        stage = str(error.stage or "").strip().lower()
        if stage == "schema":
            return (
                "LLM output did not satisfy the required "
                "analysis schema."
            )
        if stage == "json_parse":
            return "LLM output returned invalid JSON format."
        return "LLM output failed validation."

    if isinstance(error, (json.JSONDecodeError, ValidationError, ValueError, TypeError)):
        return "Insight synthesis output could not be validated against the required schema."

    return "Insight synthesis could not be completed due to an internal validation failure."


def _ensure_conditional_recommendations(
    payload: FinalInsightResponse,
    *,
    conditional_required: bool,
) -> FinalInsightResponse:
    if not conditional_required:
        return payload

    recommendations = payload.strategic_recommendations

    def _tag(items: list[str]) -> list[str]:
        out: list[str] = []
        for item in items:
            text = str(item or "").strip()
            if not text:
                continue
            if text.lower().startswith("conditional:"):
                out.append(text)
            else:
                out.append(f"Conditional: {text}")
        return out

    return payload.model_copy(
        update={
            "strategic_recommendations": recommendations.model_copy(
                update={
                    "immediate_actions": _tag(recommendations.immediate_actions),
                    "mid_term_moves": _tag(recommendations.mid_term_moves),
                    "defensive_strategies": _tag(recommendations.defensive_strategies),
                    "offensive_strategies": _tag(recommendations.offensive_strategies),
                }
            )
        }
    )


def _ensure_low_confidence_tone(
    payload: FinalInsightResponse,
    *,
    conditional_required: bool,
) -> FinalInsightResponse:
    if not conditional_required:
        return payload

    analysis = payload.competitive_analysis

    def _tag(text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return "Conditional: analysis context remains uncertain due to limited data."
        return value if value.lower().startswith("conditional:") else f"Conditional: {value}"

    return payload.model_copy(
        update={
            "competitive_analysis": analysis.model_copy(
                update={
                    "summary": _tag(analysis.summary),
                    "market_position": _tag(analysis.market_position),
                    "relative_performance": _tag(analysis.relative_performance),
                }
            )
        }
    )


def _has_competitor_data(state: AgentState) -> bool:
    """Read the deterministic competitive context contract from state.

    Returns True only when ``state["competitive_context"]["available"]``
    is explicitly ``True``.  This flag is emitted by the segmentation
    node after counting distinct peer entities in the local benchmark
    query — no heuristics, no inference, no prompt inspection.
    """
    ctx = state.get("competitive_context")
    if not isinstance(ctx, dict):
        return False
    return bool(ctx.get("available", False))


def llm_node(state: AgentState) -> AgentState:
    """LangGraph node: synthesize available outputs into a structured insight."""
    load_env_files()

    kpi_data = _usable_payload(_resolve_kpi_result(state))
    forecast_data = _usable_payload(state.get("forecast_data"))
    risk_data = _usable_payload(state.get("risk_data"))
    root_cause = _usable_payload(state.get("root_cause"))
    pipeline_status = _derive_pipeline_status(state)
    diagnostics = _collect_diagnostics(state)
    logger.info("Signal integrity scores: %s", json.dumps(diagnostics.get("signal_integrity_scores") or {}, sort_keys=True))

    has_competitors = _has_competitor_data(state)
    set_self_analysis_mode(not has_competitors)

    ctx = state.get("competitive_context") or {}
    ctx_source = str(ctx.get("source", "unavailable"))
    logger.info(
        "Competitive context: available=%s source=%s peer_count=%s metrics=%s mode=%s",
        ctx.get("available", False),
        ctx_source,
        ctx.get("peer_count", 0),
        ctx.get("metrics", []),
        "competitor" if has_competitors else "self_analysis",
    )

    # ── Extract numeric-only competitive signals for the prompt ────
    # Only structured numeric data reaches the LLM; no raw web text.
    competitor_signals: dict[str, Any] | None = None
    if has_competitors:
        numeric_signals = ctx.get("numeric_signals", [])
        if isinstance(numeric_signals, list) and numeric_signals:
            competitor_signals = {
                "source": ctx_source,
                "peer_count": ctx.get("peer_count", 0),
                "peers": ctx.get("peers", []),
                "signals": numeric_signals,
            }

    # ── Confidence penalty for external_fetch source ───────────────
    # External web-sourced competitive data is inherently less reliable
    # than deterministic local computation.  Apply a confidence penalty.
    confidence_score = float(diagnostics.get("confidence_score", 1.0))
    if ctx_source == "external_fetch" and has_competitors:
        _EXTERNAL_CONFIDENCE_PENALTY = -0.15
        confidence_score = max(0.0, round(confidence_score + _EXTERNAL_CONFIDENCE_PENALTY, 6))
        diagnostics["confidence_adjustments"].append({
            "signal": "competitive_context",
            "delta": _EXTERNAL_CONFIDENCE_PENALTY,
            "reason": "external_fetch_confidence_penalty",
        })
        diagnostics["confidence_score"] = confidence_score
        logger.info(
            "Applied external_fetch confidence penalty: delta=%.2f new_score=%.4f",
            _EXTERNAL_CONFIDENCE_PENALTY,
            confidence_score,
        )

    prompt = _prompt_builder.build_prompt(
        kpi_data=kpi_data,
        forecast_data=forecast_data,
        risk_data=risk_data,
        root_cause=root_cause,
        segmentation=_usable_payload(state.get("segmentation")),
        prioritization=state.get("prioritization") or {},
        confidence_score=confidence_score,
        missing_signals=diagnostics.get("missing_signal", []),
        has_competitor_data=has_competitors,
        competitor_signals=competitor_signals,
    )

    adapter = _build_adapter()

    try:
        synthesis = generate_with_retry(adapter, prompt)
        final_payload = FinalInsightResponse.model_validate(synthesis.model_dump())
    except Exception as error:  # noqa: BLE001
        logger.warning("LLM synthesis failed; returning structured fallback.", exc_info=True)
        final_payload = FinalInsightResponse.failure(
            reason=_public_failure_reason(error),
            pipeline_status=pipeline_status,
        )

    # Enforce: deterministic confidence always overrides LLM self-assessment.
    # The LLM must not inflate confidence beyond what the signals support.
    deterministic_confidence = float(diagnostics.get("confidence_score", 1.0))
    enforced_confidence = min(
        final_payload.competitive_analysis.confidence,
        deterministic_confidence,
    )

    # Post-processing: enforce deterministic confidence and tone rules.
    # The LLM output was already schema-validated by generate_with_retry;
    # model_copy only touches the confidence value and conditional labels,
    # so a full re-validation is not needed and would risk false-negative
    # fallback (confidence → 0) due to context-dependent validator state.
    final_payload = final_payload.model_copy(
        update={
            "competitive_analysis": final_payload.competitive_analysis.model_copy(
                update={"confidence": enforced_confidence}
            )
        }
    )
    final_payload = _ensure_conditional_recommendations(
        final_payload,
        conditional_required=(enforced_confidence < 0.5),
    )
    final_payload = _ensure_low_confidence_tone(
        final_payload,
        conditional_required=(enforced_confidence < 0.5),
    )
    final_payload = final_payload.model_copy(
        update={
            "competitive_analysis": final_payload.competitive_analysis.model_copy(
                update={"confidence": enforced_confidence}
            ),
        }
    )
    final_response = final_payload.model_dump_json()

    # Reset self-analysis mode to prevent leaking into subsequent calls.
    set_self_analysis_mode(False)

    return {
        **state,
        "pipeline_status": pipeline_status,
        "final_response": final_response,
        "envelope_diagnostics": diagnostics,
        "signal_integrity": diagnostics.get("signal_integrity"),
    }
