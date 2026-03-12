from __future__ import annotations

import logging

from agent.helpers.node_observability import wrap_node_with_structured_logging
from agent.nodes.node_result import failed, success


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_node_observability_logs_required_fields_for_success() -> None:
    logger = logging.getLogger("tests.node_observability.success")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = _CaptureHandler()
    logger.addHandler(handler)

    def node(state: dict) -> dict:
        return {
            **state,
            "growth_data": success({"value": 1.0}, confidence_score=0.82),
        }

    wrapped = wrap_node_with_structured_logging(
        node,
        node_name="growth_engine",
        output_key="growth_data",
        logger=logger,
    )
    wrapped({"request_id": "req-123", "entity_name": "acme"})

    assert handler.records
    record = handler.records[-1]
    structured = getattr(record, "structured", {})
    assert structured.get("request_id") == "req-123"
    assert structured.get("node_name") == "growth_engine"
    assert isinstance(structured.get("execution_time_ms"), (int, float))
    assert "growth_data" in structured.get("signals_generated", [])
    assert structured.get("confidence_scores", {}).get("growth_data") == 0.82
    assert structured.get("errors") == []


def test_node_observability_logs_errors_for_failed_envelope() -> None:
    logger = logging.getLogger("tests.node_observability.failed")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = _CaptureHandler()
    logger.addHandler(handler)

    def node(state: dict) -> dict:
        return {
            **state,
            "risk_data": failed("risk_engine_down", {"stage": "risk"}),
        }

    wrapped = wrap_node_with_structured_logging(
        node,
        node_name="risk",
        output_key="risk_data",
        logger=logger,
    )
    wrapped({"request_id": "req-err"})

    record = handler.records[-1]
    structured = getattr(record, "structured", {})
    errors = structured.get("errors", [])
    assert structured.get("request_id") == "req-err"
    assert structured.get("node_name") == "risk"
    assert errors
    assert "risk_engine_down" in errors[0]

