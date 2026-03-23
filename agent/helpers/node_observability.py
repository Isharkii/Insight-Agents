"""
agent/helpers/node_observability.py

Structured logging wrapper for LangGraph node execution.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

from agent.node_contracts import NodeContractValidationError, validate_contract_payload
from agent.nodes.node_result import payload_of, status_of

NodeState = dict[str, Any]
NodeFn = Callable[[NodeState], NodeState]

_OBS_LOGGER = logging.getLogger("observability.pipeline")

# Nodes that MUST succeed — failures here are fatal to the pipeline.
_CRITICAL_NODES: frozenset[str] = frozenset({
    "intent",
    "business_router",
})


def _build_degraded_envelope(
    node_name: str,
    output_key: str | None,
    error_msg: str,
) -> NodeState:
    """Build a gracefully degraded return value for a failed non-critical node."""
    if not output_key:
        return {}

    # For envelope-style keys (kpi_data, forecast_data, etc.) return a failed envelope.
    # For scalar keys (pipeline_status, prioritization, etc.) return a safe default.
    # NOTE: final_response is deliberately excluded — returning None would
    # overwrite a valid blocked response set by synthesis_gate.  If the llm
    # node degrades, returning {} preserves whatever final_response already
    # exists in state.
    _SCALAR_DEFAULTS: dict[str, Any] = {
        "pipeline_status": "partial",
        "synthesis_blocked": False,
        "prioritization": {
            "priority_level": "low",
            "recommended_focus": f"node {node_name} degraded: {error_msg[:120]}",
            "confidence_score": 0.0,
        },
    }
    if output_key == "final_response":
        # Never overwrite final_response with None on degradation.
        return {}
    if output_key in _SCALAR_DEFAULTS:
        return {output_key: _SCALAR_DEFAULTS[output_key]}

    return {
        output_key: {
            "status": "failed",
            "payload": {"error": error_msg[:500]},
            "warnings": [f"Node {node_name} degraded gracefully."],
            "errors": [error_msg[:500]],
            "confidence_score": 0.0,
        }
    }


@dataclass(frozen=True)
class NodeExecutionLog:
    """Canonical node execution log shape for observability."""

    request_id: str
    node_name: str
    execution_time_ms: float
    signals_generated: list[str]
    confidence_scores: dict[str, float]
    errors: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "node_name": self.node_name,
            "execution_time_ms": round(float(self.execution_time_ms), 3),
            "signals_generated": list(self.signals_generated),
            "confidence_scores": dict(self.confidence_scores),
            "errors": list(self.errors),
        }


def wrap_node_with_structured_logging(
    node_fn: NodeFn,
    *,
    node_name: str,
    output_key: str | None = None,
    logger: logging.Logger | None = None,
    input_contract: type[BaseModel] | None = None,
    output_contract: type[BaseModel] | None = None,
) -> NodeFn:
    """Wrap a LangGraph node and emit one structured log per execution.

    Non-critical nodes degrade gracefully on failure instead of crashing the
    entire pipeline.  Critical nodes (intent, business_router) still raise.
    """
    node_label = str(node_name or "").strip() or "unknown_node"
    out_key = str(output_key or "").strip() or None
    sink = logger or _OBS_LOGGER
    is_critical = node_label in _CRITICAL_NODES

    def _wrapped(state: NodeState) -> NodeState:
        started = time.perf_counter()
        before_state = state if isinstance(state, dict) else {}
        request_id = _request_id_from_state(before_state)

        try:
            validate_contract_payload(
                input_contract,
                before_state,
                stage="input",
                node_name=node_label,
            )
            result = node_fn(state)
            after_state_for_validation = result if isinstance(result, dict) else {}
            validate_contract_payload(
                output_contract,
                after_state_for_validation,
                stage="output",
                node_name=node_label,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            error_msg = f"{type(exc).__name__}: {exc}"
            event = NodeExecutionLog(
                request_id=request_id,
                node_name=node_label,
                execution_time_ms=elapsed_ms,
                signals_generated=[],
                confidence_scores={},
                errors=[error_msg],
            )
            sink.error(
                "node_execution_failed",
                extra={"structured": {"event": "node_execution", **event.as_dict()}},
            )

            # Critical nodes crash the pipeline; non-critical degrade gracefully.
            if is_critical:
                raise

            sink.warning(
                "node_degraded_gracefully node=%s error=%s",
                node_label,
                error_msg[:200],
            )
            return _build_degraded_envelope(node_label, out_key, error_msg)

        after_state = result if isinstance(result, dict) else {}
        request_id = _request_id_from_state(after_state) or request_id
        generated = _signals_generated(before_state, after_state, output_key=out_key)
        confidence_scores = _confidence_scores(after_state, generated)
        errors = _extract_errors(after_state, generated)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        event = NodeExecutionLog(
            request_id=request_id,
            node_name=node_label,
            execution_time_ms=elapsed_ms,
            signals_generated=generated,
            confidence_scores=confidence_scores,
            errors=errors,
        )
        payload = {"event": "node_execution", **event.as_dict()}
        if errors:
            sink.warning("node_execution", extra={"structured": payload})
        else:
            sink.info("node_execution", extra={"structured": payload})
        return result

    return _wrapped


def _request_id_from_state(state: NodeState) -> str:
    value = str(state.get("request_id") or "").strip() if isinstance(state, dict) else ""
    return value or "unknown"


def _signals_generated(
    before_state: NodeState,
    after_state: NodeState,
    *,
    output_key: str | None,
) -> list[str]:
    before_keys = set(before_state.keys())
    after_keys = set(after_state.keys())

    generated = set(after_keys - before_keys)
    for key in (before_keys & after_keys):
        if before_state.get(key) != after_state.get(key):
            generated.add(key)

    if output_key and output_key in after_keys:
        generated.add(output_key)

    return sorted(str(item) for item in generated)


def _confidence_scores(state: NodeState, generated: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key in generated:
        value = state.get(key)
        score = _extract_confidence(value)
        if score is not None:
            scores[key] = round(score, 6)
    return scores


def _extract_confidence(value: Any) -> float | None:
    if not isinstance(value, dict):
        return None

    raw = value.get("confidence_score")
    if isinstance(raw, (int, float)):
        return float(raw)

    payload = value.get("payload")
    if isinstance(payload, dict):
        nested = payload.get("confidence_score")
        if isinstance(nested, (int, float)):
            return float(nested)
    return None


def _extract_errors(state: NodeState, generated: list[str]) -> list[str]:
    errors: list[str] = []
    for key in generated:
        value = state.get(key)
        if not isinstance(value, dict):
            continue

        raw_errors = value.get("errors")
        if isinstance(raw_errors, list):
            errors.extend(str(item) for item in raw_errors if str(item).strip())

        status = status_of(value)
        if status != "failed":
            continue
        payload = payload_of(value) or {}
        reason = payload.get("error") or payload.get("reason")
        if reason:
            errors.append(str(reason))
        else:
            errors.append(f"{key}:failed")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in errors:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped
