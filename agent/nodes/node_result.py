"""
agent/nodes/node_result.py

Shared helpers for node-level status/payload envelopes.

Standard envelope statuses are:
  - ``"success"``            — node ran, produced a valid result.
  - ``"insufficient_data"``  — node ran, data was too shallow or
                                metrics were missing.  The node
                                produced a best-effort payload that
                                downstream consumers may use with
                                degraded confidence.  NOT a failure.
  - ``"skipped"``            — node did not run (pre-condition unmet).
  - ``"failed"``             — node ran but hit an unrecoverable error.

Envelopes may optionally include ``warnings``, ``errors``, and
``confidence_score`` to preserve degraded-signal diagnostics without
introducing a separate status class.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

NodeStatus = Literal["success", "insufficient_data", "skipped", "failed"]

_VALID_STATUSES = {"success", "insufficient_data", "skipped", "failed"}
_LEGACY_PARTIAL_STATUS = "partial"


def success(
    payload: dict[str, Any] | None,
    *,
    warnings: Sequence[str] = (),
    errors: Sequence[str] = (),
    confidence_score: float = 1.0,
) -> dict[str, Any]:
    """Build a success envelope with optional diagnostics."""
    return {
        "status": "success",
        "payload": payload,
        "warnings": list(warnings),
        "errors": list(errors),
        "confidence_score": max(0.0, min(1.0, confidence_score)),
    }


def partial(
    payload: dict[str, Any] | None,
    *,
    warnings: Sequence[str] = (),
    errors: Sequence[str] = (),
    confidence_score: float = 0.5,
) -> dict[str, Any]:
    """Backward-compatible alias that now emits ``status='success'``."""
    return success(
        payload,
        warnings=warnings,
        errors=errors,
        confidence_score=confidence_score,
    )


def insufficient_data(
    reason: str,
    payload: dict[str, Any] | None = None,
    *,
    warnings: Sequence[str] = (),
    confidence_score: float = 0.3,
) -> dict[str, Any]:
    """Build an insufficient_data envelope.

    The node executed but the input data was too shallow (e.g. < 2
    time-series periods) or required metrics were absent.  A
    best-effort payload is included so downstream consumers can
    degrade gracefully rather than treating this as a hard failure.
    """
    body = payload or {}
    if reason:
        body.setdefault("reason", reason)
    return {
        "status": "insufficient_data",
        "payload": body or None,
        "warnings": list(warnings),
        "confidence_score": max(0.0, min(1.0, confidence_score)),
    }


def skipped(reason: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = payload or {}
    if reason:
        body.setdefault("reason", reason)
    return {"status": "skipped", "payload": body or None}


def failed(error: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = payload or {}
    if error:
        body.setdefault("error", error)
    return {"status": "failed", "payload": body or None}


def status_of(value: Any) -> NodeStatus:
    if isinstance(value, dict):
        status = value.get("status")
        if status in _VALID_STATUSES:
            return status
        if status == _LEGACY_PARTIAL_STATUS:
            return "success"
    return "failed"


def payload_of(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        status = value.get("status")
        payload = value.get("payload")
        if status in _VALID_STATUSES or status == _LEGACY_PARTIAL_STATUS:
            return payload if isinstance(payload, dict) else None
        return value
    return None


def warnings_of(value: Any) -> list[str]:
    """Extract warnings list from an envelope (empty if absent)."""
    if isinstance(value, dict):
        w = value.get("warnings")
        if isinstance(w, list):
            return [str(item) for item in w]
    return []


def errors_of(value: Any) -> list[str]:
    """Extract errors list from an envelope (empty if absent)."""
    if isinstance(value, dict):
        e = value.get("errors")
        if isinstance(e, list):
            return [str(item) for item in e]
    return []


def confidence_of(value: Any) -> float:
    """Extract confidence_score from an envelope (0.0 if absent)."""
    if isinstance(value, dict):
        score = value.get("confidence_score")
        if isinstance(score, (int, float)):
            return float(score)
    return 0.0
