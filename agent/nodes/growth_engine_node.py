"""
agent/nodes/growth_engine_node.py

Compute deterministic growth signals from KPI time series.
"""

from __future__ import annotations

from agent.helpers.confidence_model import compute_standard_confidence
from agent.helpers.kpi_extraction import (
    dataset_confidence_from_state,
    metric_series_from_kpi_payload,
    records_from_kpi_payload,
    resolve_kpi_payload,
)
from agent.nodes.node_result import failed, skipped, success
from agent.state import AgentState
from app.services.kpi_canonical_schema import metric_aliases_for_business_type
from app.services.statistics.growth_engine import compute_growth_context


def growth_engine_node(state: AgentState) -> AgentState:
    """Compute deterministic growth signals from KPI time series."""
    try:
        kpi_payload = resolve_kpi_payload(state)
        records = records_from_kpi_payload(kpi_payload)
        metric_series = metric_series_from_kpi_payload(kpi_payload)
        if not metric_series:
            return {
                "growth_data": skipped(
                    "series_unavailable",
                    {"records": len(records), "record_count": int(kpi_payload.get("record_count") or 0)},
                ),
            }

        business_type = str(state.get("business_type") or "").strip().lower()
        aliases = metric_aliases_for_business_type(business_type)
        revenue_candidates = aliases.get("recurring_revenue", ("recurring_revenue",))
        growth_context = compute_growth_context(
            metric_series,
            preferred_metric_candidates=revenue_candidates,
        )

        dataset_confidence = dataset_confidence_from_state(state)
        warnings = [str(item) for item in growth_context.get("warnings", [])]
        if dataset_confidence < 1.0:
            warnings.append(
                f"Dataset confidence reduced growth reliability ({dataset_confidence:.2f})."
            )
        primary_metric = str(growth_context.get("primary_metric") or "").strip()
        if primary_metric not in metric_series and metric_series:
            primary_metric = sorted(
                metric_series,
                key=lambda name: (-len(metric_series[name]), name),
            )[0]
        primary_values = metric_series.get(primary_metric, []) if primary_metric else []
        horizons = growth_context.get("primary_horizons")
        if not isinstance(horizons, dict):
            horizons = {}
        confidence_model = compute_standard_confidence(
            values=primary_values,
            signals={
                "short_growth": horizons.get("short_growth"),
                "mid_growth": horizons.get("mid_growth"),
                "long_growth": horizons.get("long_growth"),
                "trend_acceleration": horizons.get("trend_acceleration"),
                "cagr": horizons.get("cagr"),
            },
            dataset_confidence=dataset_confidence,
            upstream_confidences=[float(growth_context.get("confidence_score") or 0.0)],
            status=str(growth_context.get("status") or "success"),
            base_warnings=warnings,
        )
        growth_payload = {
            **growth_context,
            "confidence_breakdown": confidence_model,
        }
        growth_data = success(
            growth_payload,
            warnings=confidence_model["warnings"],
            confidence_score=float(confidence_model["confidence_score"]),
        )
        return {"growth_data": growth_data}
    except Exception as exc:  # noqa: BLE001
        return {"growth_data": failed(str(exc), {"stage": "growth_engine"})}
