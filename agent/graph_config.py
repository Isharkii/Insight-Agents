"""
agent/graph_config.py

Central graph-node configuration for required vs optional signal nodes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphNodeConfig:
    required: tuple[str, ...]
    optional: tuple[str, ...]


KPI_KEY_BY_BUSINESS_TYPE: dict[str, str] = {
    "saas": "saas_kpi_data",
    "ecommerce": "ecommerce_kpi_data",
    "agency": "agency_kpi_data",
    "general_timeseries": "kpi_data",
    "generic_timeseries": "kpi_data",
}


NODE_CONFIG_BY_BUSINESS_TYPE: dict[str, GraphNodeConfig] = {
    "saas": GraphNodeConfig(
        required=("saas_kpi_data", "risk_data"),
        optional=(
            "growth_data",
            "timeseries_factors_data",
            "cohort_data",
            "category_formula_data",
            "unit_economics_data",
            "multivariate_scenario_data",
            "segmentation",
            "forecast_data",
            "signal_conflicts",
            "root_cause",
        ),
    ),
    "ecommerce": GraphNodeConfig(
        required=("ecommerce_kpi_data", "risk_data"),
        optional=(
            "growth_data",
            "timeseries_factors_data",
            "cohort_data",
            "category_formula_data",
            "unit_economics_data",
            "multivariate_scenario_data",
            "segmentation",
            "forecast_data",
            "signal_conflicts",
            "root_cause",
        ),
    ),
    "agency": GraphNodeConfig(
        required=("agency_kpi_data", "risk_data"),
        optional=(
            "growth_data",
            "timeseries_factors_data",
            "cohort_data",
            "category_formula_data",
            "unit_economics_data",
            "multivariate_scenario_data",
            "segmentation",
            "forecast_data",
            "signal_conflicts",
            "root_cause",
        ),
    ),
    "general_timeseries": GraphNodeConfig(
        required=("kpi_data",),
        optional=(
            "growth_data",
            "timeseries_factors_data",
            "cohort_data",
            "category_formula_data",
            "unit_economics_data",
            "multivariate_scenario_data",
            "segmentation",
            "risk_data",
            "forecast_data",
            "signal_conflicts",
            "root_cause",
        ),
    ),
    "generic_timeseries": GraphNodeConfig(
        required=("kpi_data",),
        optional=(
            "growth_data",
            "timeseries_factors_data",
            "cohort_data",
            "category_formula_data",
            "unit_economics_data",
            "multivariate_scenario_data",
            "segmentation",
            "risk_data",
            "forecast_data",
            "signal_conflicts",
            "root_cause",
        ),
    ),
}


def normalize_business_type(value: str | None) -> str:
    return str(value or "").strip().lower()


def graph_node_config_for_business_type(business_type: str | None) -> GraphNodeConfig:
    normalized = normalize_business_type(business_type)
    if normalized in NODE_CONFIG_BY_BUSINESS_TYPE:
        return NODE_CONFIG_BY_BUSINESS_TYPE[normalized]
    return GraphNodeConfig(
        required=("kpi_data",),
        optional=(
            "growth_data",
            "timeseries_factors_data",
            "cohort_data",
            "category_formula_data",
            "unit_economics_data",
            "multivariate_scenario_data",
            "segmentation",
            "risk_data",
            "forecast_data",
            "signal_conflicts",
            "root_cause",
        ),
    )


def signal_name_for_state_key(state_key: str) -> str:
    text = str(state_key or "").strip().lower()
    if text.endswith("_data"):
        text = text[:-5]
    return text
