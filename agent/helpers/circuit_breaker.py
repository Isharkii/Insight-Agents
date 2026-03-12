"""
agent/helpers/circuit_breaker.py

Reusable circuit breaker for LangGraph node execution.

This utility wraps node callables, tracks repeated failures, opens the
circuit after a threshold, and returns degraded envelopes while open.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Literal

from agent.nodes.node_result import insufficient_data, payload_of, status_of

CircuitState = Literal["closed", "open", "half_open"]
NodeState = dict[str, Any]
NodeFn = Callable[[NodeState], NodeState]
FailurePredicate = Callable[[NodeState], bool]
DegradeFn = Callable[[NodeState, "CircuitBreakerSnapshot", str], NodeState]


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Config for node-level circuit breaker behavior."""

    failure_threshold: int = 3
    cooldown_seconds: float = 60.0
    half_open_max_calls: int = 1
    degraded_confidence: float = 0.2

    def __post_init__(self) -> None:
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be > 0")
        if self.cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be > 0")
        if self.half_open_max_calls <= 0:
            raise ValueError("half_open_max_calls must be > 0")
        if not 0.0 <= float(self.degraded_confidence) <= 1.0:
            raise ValueError("degraded_confidence must be between 0 and 1")


@dataclass(frozen=True)
class CircuitBreakerSnapshot:
    """Immutable point-in-time breaker state."""

    name: str
    state: CircuitState
    failure_threshold: int
    cooldown_seconds: float
    consecutive_failures: int
    remaining_cooldown_seconds: float
    open_events: int
    last_error: str | None


class CircuitBreaker:
    """Thread-safe circuit breaker instance for one logical node."""

    def __init__(
        self,
        *,
        name: str,
        config: CircuitBreakerConfig,
        now_fn: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
    ) -> None:
        self._name = str(name or "").strip() or "unnamed_node"
        self._config = config
        self._now_fn = now_fn
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()

        self._state: CircuitState = "closed"
        self._consecutive_failures = 0
        self._opened_until = 0.0
        self._half_open_calls = 0
        self._open_events = 0
        self._last_error: str | None = None

    def allow_request(self) -> tuple[bool, CircuitBreakerSnapshot]:
        """Return whether a wrapped call may execute right now."""
        with self._lock:
            now = self._now_fn()
            if self._state == "open":
                if now < self._opened_until:
                    return False, self._snapshot_locked(now)
                self._state = "half_open"
                self._half_open_calls = 0
                self._logger.info(
                    "circuit_half_open name=%s cooldown_seconds=%.2f",
                    self._name,
                    self._config.cooldown_seconds,
                )

            if self._state == "half_open":
                if self._half_open_calls >= self._config.half_open_max_calls:
                    return False, self._snapshot_locked(now)
                self._half_open_calls += 1

            return True, self._snapshot_locked(now)

    def record_success(self) -> CircuitBreakerSnapshot:
        """Record a successful execution and close/reset the circuit."""
        with self._lock:
            now = self._now_fn()
            prev_state = self._state
            self._state = "closed"
            self._consecutive_failures = 0
            self._opened_until = 0.0
            self._half_open_calls = 0
            self._last_error = None
            if prev_state != "closed":
                self._logger.info("circuit_closed name=%s previous_state=%s", self._name, prev_state)
            return self._snapshot_locked(now)

    def record_failure(self, error: str | None = None) -> CircuitBreakerSnapshot:
        """Record a failure and open the circuit if threshold is reached."""
        with self._lock:
            now = self._now_fn()
            message = str(error or "").strip() or "unknown_error"
            self._last_error = message

            if self._state == "half_open":
                self._trip_open_locked(now, reason="half_open_failure")
                return self._snapshot_locked(now)

            self._consecutive_failures += 1
            if self._consecutive_failures >= self._config.failure_threshold:
                self._trip_open_locked(now, reason="failure_threshold_reached")
            else:
                self._logger.warning(
                    "circuit_failure name=%s failures=%d/%d error=%s",
                    self._name,
                    self._consecutive_failures,
                    self._config.failure_threshold,
                    message,
                )
            return self._snapshot_locked(now)

    def snapshot(self) -> CircuitBreakerSnapshot:
        with self._lock:
            return self._snapshot_locked(self._now_fn())

    def _trip_open_locked(self, now: float, *, reason: str) -> None:
        self._state = "open"
        self._opened_until = now + self._config.cooldown_seconds
        self._consecutive_failures = 0
        self._half_open_calls = 0
        self._open_events += 1
        self._logger.warning(
            "circuit_opened name=%s reason=%s cooldown_seconds=%.2f open_events=%d last_error=%s",
            self._name,
            reason,
            self._config.cooldown_seconds,
            self._open_events,
            self._last_error,
        )

    def _snapshot_locked(self, now: float) -> CircuitBreakerSnapshot:
        remaining = 0.0
        if self._state == "open":
            remaining = max(0.0, self._opened_until - now)
        return CircuitBreakerSnapshot(
            name=self._name,
            state=self._state,
            failure_threshold=self._config.failure_threshold,
            cooldown_seconds=self._config.cooldown_seconds,
            consecutive_failures=self._consecutive_failures,
            remaining_cooldown_seconds=round(remaining, 6),
            open_events=self._open_events,
            last_error=self._last_error,
        )


class CircuitBreakerRegistry:
    """Named circuit breaker store for reusable node wrapping."""

    def __init__(
        self,
        *,
        default_config: CircuitBreakerConfig | None = None,
        now_fn: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
    ) -> None:
        self._default_config = default_config or CircuitBreakerConfig()
        self._now_fn = now_fn
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        name: str,
        *,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        key = str(name or "").strip() or "unnamed_node"
        with self._lock:
            existing = self._breakers.get(key)
            if existing is not None:
                return existing
            breaker = CircuitBreaker(
                name=key,
                config=config or self._default_config,
                now_fn=self._now_fn,
                logger=self._logger,
            )
            self._breakers[key] = breaker
            return breaker

    def reset(self, name: str | None = None) -> None:
        with self._lock:
            if name is None:
                self._breakers.clear()
                return
            self._breakers.pop(str(name), None)


GLOBAL_NODE_CIRCUIT_BREAKERS = CircuitBreakerRegistry()


def envelope_failure_predicate(output_key: str) -> FailurePredicate:
    """Failure predicate for node_result envelope outputs."""

    key = str(output_key or "").strip()

    def _predicate(state: NodeState) -> bool:
        return status_of(state.get(key)) == "failed"

    return _predicate


def wrap_langgraph_node(
    node_fn: NodeFn,
    *,
    node_name: str,
    output_key: str | None = None,
    config: CircuitBreakerConfig | None = None,
    registry: CircuitBreakerRegistry | None = None,
    failure_predicate: FailurePredicate | None = None,
    degrade_fn: DegradeFn | None = None,
    degrade_on_failure: bool = False,
) -> NodeFn:
    """Wrap a LangGraph node with circuit breaker semantics."""

    breaker_registry = registry or GLOBAL_NODE_CIRCUIT_BREAKERS
    breaker = breaker_registry.get_or_create(node_name, config=config)
    out_key = str(output_key or "").strip() or None

    if failure_predicate is None:
        failure_predicate = envelope_failure_predicate(out_key) if out_key else (lambda _state: False)
    if degrade_fn is None:
        degrade_fn = lambda state, snapshot, reason: _default_degrade_state(  # noqa: E731
            state,
            node_name=node_name,
            output_key=out_key,
            snapshot=snapshot,
            reason=reason,
            degraded_confidence=(config.degraded_confidence if config else 0.2),
        )

    @wraps(node_fn)
    def _wrapped(state: NodeState) -> NodeState:
        allowed, snapshot = breaker.allow_request()
        if not allowed:
            return degrade_fn(state, snapshot, "circuit_breaker_open")

        try:
            result = node_fn(state)
        except Exception as exc:  # noqa: BLE001
            failed_snapshot = breaker.record_failure(str(exc))
            return degrade_fn(state, failed_snapshot, "node_exception")

        if failure_predicate(result):
            reason = _extract_failure_reason(result, output_key=out_key)
            failed_snapshot = breaker.record_failure(reason)
            if degrade_on_failure:
                return degrade_fn(result, failed_snapshot, "node_failed")
            return result

        breaker.record_success()
        return result

    return _wrapped


def _extract_failure_reason(state: NodeState, *, output_key: str | None) -> str:
    if not output_key:
        return "failure_predicate_triggered"

    envelope = state.get(output_key)
    payload = payload_of(envelope) or {}
    reason = payload.get("error") or payload.get("reason")
    if reason:
        return str(reason)
    return f"{output_key}:{status_of(envelope)}"


def _default_degrade_state(
    state: NodeState,
    *,
    node_name: str,
    output_key: str | None,
    snapshot: CircuitBreakerSnapshot,
    reason: str,
    degraded_confidence: float,
) -> NodeState:
    warning = (
        f"{node_name} degraded via circuit breaker ({reason}); "
        f"state={snapshot.state}; retry_after={snapshot.remaining_cooldown_seconds:.1f}s."
    )
    metadata = {
        "name": snapshot.name,
        "state": snapshot.state,
        "open_events": snapshot.open_events,
        "remaining_cooldown_seconds": snapshot.remaining_cooldown_seconds,
        "last_error": snapshot.last_error,
    }

    if output_key:
        existing_payload = payload_of(state.get(output_key)) or {}
        payload = {**existing_payload, "circuit_breaker": metadata}
        degraded = insufficient_data(
            reason="circuit_breaker_open" if snapshot.state == "open" else reason,
            payload=payload,
            warnings=[warning],
            confidence_score=degraded_confidence,
        )
        return {**state, output_key: degraded}

    diagnostics = state.get("envelope_diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    warnings = diagnostics.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    warnings = [*warnings, warning]
    diagnostics = {**diagnostics, "warnings": warnings, "circuit_breaker": metadata}
    return {**state, "envelope_diagnostics": diagnostics}

