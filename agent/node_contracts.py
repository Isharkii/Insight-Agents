"""
agent/node_contracts.py

Pydantic-backed node input/output contracts for LangGraph nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from agent.state_models import (
    AgentStatePatchModel,
    CompetitiveContextModel,
    ForecastPayloadModel,
    KpiPayloadModel,
    NodeEnvelope,
    PipelineStatus,
    RiskPayloadModel,
)


class NodeContractValidationError(ValueError):
    """Raised when a node input/output violates a typed contract."""


@dataclass(frozen=True)
class NodeContractSpec:
    """Contract pair used to validate node execution."""

    input_model: type[BaseModel]
    output_model: type[BaseModel] | None


class QueryInputContract(AgentStatePatchModel):
    model_config = ConfigDict(extra="forbid")
    user_query: str


class BusinessTypeInputContract(AgentStatePatchModel):
    model_config = ConfigDict(extra="forbid")
    business_type: str


class EntityBusinessInputContract(AgentStatePatchModel):
    model_config = ConfigDict(extra="forbid")
    entity_name: str
    business_type: str


class EntityInputContract(AgentStatePatchModel):
    model_config = ConfigDict(extra="forbid")
    entity_name: str


class PipelineStatusOutputContract(BaseModel):
    model_config = ConfigDict(extra="allow")
    pipeline_status: PipelineStatus


class FinalResponseOutputContract(BaseModel):
    model_config = ConfigDict(extra="allow")
    final_response: str


class SynthesisBlockedOutputContract(BaseModel):
    model_config = ConfigDict(extra="allow")
    synthesis_blocked: bool


class CompetitiveContextOutputContract(BaseModel):
    model_config = ConfigDict(extra="allow")
    competitive_context: CompetitiveContextModel


class PrioritizationOutputContract(BaseModel):
    model_config = ConfigDict(extra="allow")
    prioritization: dict[str, Any]


_ENVELOPE_PAYLOAD_BY_OUTPUT_KEY: dict[str, type[Any]] = {
    "kpi_data": KpiPayloadModel,
    "saas_kpi_data": KpiPayloadModel,
    "ecommerce_kpi_data": KpiPayloadModel,
    "agency_kpi_data": KpiPayloadModel,
    "forecast_data": ForecastPayloadModel,
    "risk_data": RiskPayloadModel | dict[str, Any],
    "root_cause": dict[str, Any],
    "segmentation": dict[str, Any],
    "growth_data": dict[str, Any],
    "timeseries_factors_data": dict[str, Any],
    "cohort_data": dict[str, Any],
    "category_formula_data": dict[str, Any],
    "unit_economics_data": dict[str, Any],
    "multivariate_scenario_data": dict[str, Any],
    "signal_conflicts": dict[str, Any],
    "signal_enrichment": dict[str, Any],
}

_SCALAR_OUTPUT_CONTRACT_BY_KEY: dict[str, type[BaseModel]] = {
    "pipeline_status": PipelineStatusOutputContract,
    "final_response": FinalResponseOutputContract,
    "synthesis_blocked": SynthesisBlockedOutputContract,
    "competitive_context": CompetitiveContextOutputContract,
    "prioritization": PrioritizationOutputContract,
}

_INPUT_CONTRACT_BY_NODE: dict[str, type[BaseModel]] = {
    "intent": QueryInputContract,
    "business_router": BusinessTypeInputContract,
    "kpi_fetch": EntityBusinessInputContract,
    "saas_kpi_fetch": EntityBusinessInputContract,
    "ecommerce_kpi_fetch": EntityBusinessInputContract,
    "agency_kpi_fetch": EntityBusinessInputContract,
    "forecast_fetch": EntityBusinessInputContract,
    "risk": EntityInputContract,
    "prioritization": EntityInputContract,
    "competitor_intelligence": EntityInputContract,
}


def validate_contract_payload(
    model: type[BaseModel] | None,
    payload: dict[str, Any],
    *,
    stage: Literal["input", "output"],
    node_name: str,
) -> None:
    if model is None:
        return
    try:
        model.model_validate(payload)
    except ValidationError as exc:
        raise NodeContractValidationError(
            f"{stage} contract validation failed for node={node_name}: {exc}"
        ) from exc


def input_contract_for_node(node_name: str) -> type[BaseModel]:
    return _INPUT_CONTRACT_BY_NODE.get(node_name, AgentStatePatchModel)


@lru_cache(maxsize=64)
def output_contract_for_key(output_key: str | None) -> type[BaseModel] | None:
    key = str(output_key or "").strip()
    if not key:
        return None

    scalar_contract = _SCALAR_OUTPUT_CONTRACT_BY_KEY.get(key)
    if scalar_contract is not None:
        return scalar_contract

    payload_type = _ENVELOPE_PAYLOAD_BY_OUTPUT_KEY.get(key, dict[str, Any])
    envelope_type = NodeEnvelope[payload_type]  # type: ignore[valid-type]
    model_name = "".join(part.capitalize() for part in key.split("_")) + "OutputContract"
    return create_model(
        model_name,
        __config__=ConfigDict(extra="allow"),
        **{key: (envelope_type, ...)},
    )


def contract_spec_for_node(
    *,
    node_name: str,
    output_key: str | None,
) -> NodeContractSpec:
    return NodeContractSpec(
        input_model=input_contract_for_node(node_name),
        output_model=output_contract_for_key(output_key),
    )

