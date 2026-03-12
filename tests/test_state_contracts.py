from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from agent.helpers.node_observability import wrap_node_with_structured_logging
from agent.node_contracts import (
    NodeContractValidationError,
    input_contract_for_node,
    output_contract_for_key,
    validate_contract_payload,
)
from agent.state_models import AgentStateModel


def test_agent_state_model_validates_typed_schema() -> None:
    state = AgentStateModel.model_validate(
        {
            "request_id": "req-1",
            "user_query": "How is growth trending?",
            "business_type": "saas",
            "entity_name": "acme",
            "kpi_data": {
                "status": "success",
                "payload": {
                    "state_mode": "derived_only",
                    "record_ref": "abc-123",
                    "record_count": 12,
                    "metric_series": {"mrr": [100.0, 110.0, 120.0]},
                    "latest_computed_kpis": {"mrr": {"value": 120.0}},
                },
                "warnings": [],
                "errors": [],
                "confidence_score": 0.91,
            },
        }
    )
    dumped = state.to_graph_state()
    assert dumped["entity_name"] == "acme"
    assert dumped["kpi_data"]["status"] == "success"


def test_agent_state_model_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        AgentStateModel.model_validate(
            {
                "request_id": "req-1",
                "user_query": "x",
                "business_type": "saas",
                "entity_name": "acme",
                "unexpected_key": {"foo": "bar"},
            }
        )


def test_output_contract_rejects_invalid_forecast_payload() -> None:
    output_contract = output_contract_for_key("forecast_data")
    assert output_contract is not None

    with pytest.raises(NodeContractValidationError):
        validate_contract_payload(
            output_contract,
            {
                "forecast_data": {
                    "status": "success",
                    "payload": "invalid_payload_shape",
                }
            },
            stage="output",
            node_name="forecast_fetch",
        )


def test_node_wrapper_enforces_input_output_contracts() -> None:
    """Non-critical nodes degrade gracefully on contract violation
    instead of crashing the pipeline."""
    logger = logging.getLogger("tests.state_contracts.wrapper")
    logger.handlers.clear()
    logger.propagate = False

    def bad_node(state: dict) -> dict:
        return {
            **state,
            "kpi_data": {
                "status": "success",
                "payload": "not_a_mapping",
            },
        }

    wrapped = wrap_node_with_structured_logging(
        bad_node,
        node_name="kpi_fetch",
        output_key="kpi_data",
        logger=logger,
        input_contract=input_contract_for_node("kpi_fetch"),
        output_contract=output_contract_for_key("kpi_data"),
    )

    # Non-critical nodes degrade gracefully — they return a failed envelope
    # instead of raising.
    result = wrapped(
        {
            "request_id": "req-2",
            "business_type": "saas",
            "entity_name": "acme",
        }
    )
    assert result["kpi_data"]["status"] == "failed"
    assert result["kpi_data"]["confidence_score"] == 0.0


def test_critical_node_wrapper_still_raises() -> None:
    """Critical nodes (intent, business_router) must still raise on failure."""
    logger = logging.getLogger("tests.state_contracts.critical")
    logger.handlers.clear()
    logger.propagate = False

    def bad_intent(state: dict) -> dict:
        raise ValueError("intent failed")

    wrapped = wrap_node_with_structured_logging(
        bad_intent,
        node_name="intent",
        output_key=None,
        logger=logger,
        input_contract=input_contract_for_node("intent"),
        output_contract=None,
    )

    with pytest.raises(ValueError, match="intent failed"):
        wrapped({"user_query": "test"})

