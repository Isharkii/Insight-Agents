from __future__ import annotations

import json
import logging

from app.observability.logging_config import StructuredJsonFormatter


def test_structured_json_formatter_includes_structured_payload() -> None:
    formatter = StructuredJsonFormatter()

    record = logging.LogRecord(
        name="observability.pipeline",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="node_execution",
        args=(),
        exc_info=None,
    )
    record.structured = {
        "request_id": "req-1",
        "node_name": "growth_engine",
        "execution_time_ms": 12.3,
        "signals_generated": ["growth_data"],
        "confidence_scores": {"growth_data": 0.81},
        "errors": [],
    }

    rendered = formatter.format(record)
    payload = json.loads(rendered)

    assert payload["request_id"] == "req-1"
    assert payload["node_name"] == "growth_engine"
    assert payload["execution_time_ms"] == 12.3
    assert payload["signals_generated"] == ["growth_data"]
    assert payload["confidence_scores"] == {"growth_data": 0.81}
    assert payload["errors"] == []

