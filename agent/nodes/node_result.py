"""
agent/nodes/node_result.py

Shared helpers for node-level status/payload envelopes.
"""

from __future__ import annotations

from typing import Any, Literal

NodeStatus = Literal["success", "skipped", "failed"]


def success(payload: dict[str, Any] | None) -> dict[str, Any]:
    return {"status": "success", "payload": payload}


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
        if status in {"success", "skipped", "failed"}:
            return status
    return "failed"


def payload_of(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        status = value.get("status")
        payload = value.get("payload")
        if status in {"success", "skipped", "failed"}:
            return payload if isinstance(payload, dict) else None
        return value
    return None
