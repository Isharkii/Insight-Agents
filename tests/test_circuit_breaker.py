from __future__ import annotations

from agent.helpers.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    wrap_langgraph_node,
)
from agent.nodes.node_result import failed, status_of, success


class _FakeClock:
    def __init__(self) -> None:
        self._now = 0.0

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += float(seconds)


def test_circuit_breaker_opens_at_threshold_and_recovers_after_cooldown() -> None:
    clock = _FakeClock()
    breaker = CircuitBreaker(
        name="test_node",
        config=CircuitBreakerConfig(
            failure_threshold=2,
            cooldown_seconds=30.0,
            half_open_max_calls=1,
        ),
        now_fn=clock.now,
    )

    allowed, snapshot = breaker.allow_request()
    assert allowed is True
    assert snapshot.state == "closed"

    snapshot = breaker.record_failure("first_failure")
    assert snapshot.state == "closed"

    snapshot = breaker.record_failure("second_failure")
    assert snapshot.state == "open"
    assert snapshot.remaining_cooldown_seconds >= 29.9

    allowed, snapshot = breaker.allow_request()
    assert allowed is False
    assert snapshot.state == "open"

    clock.advance(30.5)
    allowed, snapshot = breaker.allow_request()
    assert allowed is True
    assert snapshot.state == "half_open"

    snapshot = breaker.record_success()
    assert snapshot.state == "closed"


def test_wrap_langgraph_node_short_circuits_when_open() -> None:
    clock = _FakeClock()
    registry = CircuitBreakerRegistry(
        default_config=CircuitBreakerConfig(
            failure_threshold=2,
            cooldown_seconds=60.0,
        ),
        now_fn=clock.now,
    )

    calls = {"count": 0}

    def unstable_node(state: dict) -> dict:
        calls["count"] += 1
        return {**state, "forecast_data": failed("db_unavailable", {"stage": "fetch"})}

    wrapped = wrap_langgraph_node(
        unstable_node,
        node_name="forecast_fetch",
        output_key="forecast_data",
        registry=registry,
    )

    first = wrapped({})
    assert status_of(first.get("forecast_data")) == "failed"

    second = wrapped({})
    assert status_of(second.get("forecast_data")) == "failed"

    third = wrapped({})
    assert status_of(third.get("forecast_data")) == "insufficient_data"
    assert calls["count"] == 2
    assert third["forecast_data"]["payload"]["reason"] == "circuit_breaker_open"


def test_wrap_langgraph_node_retries_after_cooldown_and_closes_on_success() -> None:
    clock = _FakeClock()
    registry = CircuitBreakerRegistry(
        default_config=CircuitBreakerConfig(
            failure_threshold=2,
            cooldown_seconds=10.0,
            half_open_max_calls=1,
        ),
        now_fn=clock.now,
    )

    calls = {"count": 0}

    def flaky_then_ok_node(state: dict) -> dict:
        calls["count"] += 1
        if calls["count"] <= 2:
            return {**state, "forecast_data": failed("temporary_failure", {"attempt": calls["count"]})}
        return {**state, "forecast_data": success({"ok": True})}

    wrapped = wrap_langgraph_node(
        flaky_then_ok_node,
        node_name="forecast_fetch_retry",
        output_key="forecast_data",
        registry=registry,
    )

    wrapped({})
    wrapped({})

    open_call = wrapped({})
    assert status_of(open_call.get("forecast_data")) == "insufficient_data"
    assert calls["count"] == 2

    clock.advance(10.5)
    recovered = wrapped({})
    assert status_of(recovered.get("forecast_data")) == "success"

    breaker = registry.get_or_create("forecast_fetch_retry")
    assert breaker.snapshot().state == "closed"

