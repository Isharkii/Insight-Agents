"""
agent/nodes/multivariate_scenario_node.py

Compute statistical, multivariate, and deterministic scenario payloads.
"""

from __future__ import annotations

from agent.helpers.canonical_queries import cohort_rows_for_records
from agent.helpers.confidence_model import compute_standard_confidence
from agent.helpers.kpi_extraction import (
    dataset_confidence_from_state,
    metric_series_from_kpi_payload,
    records_from_kpi_payload,
    resolve_kpi_payload,
    resolve_period_bounds,
)
from agent.helpers.statistical_context import build_statistical_context
from agent.nodes.node_result import failed, payload_of, skipped, success
from agent.state import AgentState
from app.services.kpi_canonical_schema import metric_aliases_for_business_type
from app.services.statistics.causal_inference import granger_causality
from app.services.statistics.growth_engine import compute_growth_context
from app.services.statistics.multivariate import compute_multivariate_context
from app.services.statistics.scenario_simulator import simulate_deterministic_scenarios


def multivariate_scenario_node(state: AgentState) -> AgentState:
    """Compute statistical, multivariate, and deterministic scenario payloads."""
    try:
        kpi_payload = resolve_kpi_payload(state)
        records = records_from_kpi_payload(kpi_payload)
        if not isinstance(kpi_payload, dict):
            return {
                "multivariate_scenario_data": skipped("kpi_unavailable", {"records": 0}),
            }

        metric_series = metric_series_from_kpi_payload(kpi_payload)
        if not metric_series:
            return {
                "multivariate_scenario_data": skipped(
                    "series_unavailable",
                    {"records": len(records), "record_count": int(kpi_payload.get("record_count") or 0)},
                ),
            }

        business_type = str(state.get("business_type") or "").strip().lower()
        aliases = metric_aliases_for_business_type(business_type)
        revenue_candidates = aliases.get("recurring_revenue", ("recurring_revenue",))
        growth_context = payload_of(state.get("growth_data")) or compute_growth_context(
            metric_series,
            preferred_metric_candidates=revenue_candidates,
        )
        statistical_context = build_statistical_context(metric_series)

        entity_name = str(
            kpi_payload.get("fetched_for")
            or state.get("entity_name")
            or ""
        ).strip()
        period_start, period_end = resolve_period_bounds(kpi_payload)
        cohort_rows = cohort_rows_for_records(
            records=records,
            entity_name=entity_name,
            business_type=business_type,
            period_start=period_start,
            period_end=period_end,
        )
        multivariate_context = compute_multivariate_context(
            metric_series,
            segment_rows=cohort_rows,
            preferred_metric_candidates=revenue_candidates,
        )
        causal_context = granger_causality(metric_series)
        scenario_simulation = simulate_deterministic_scenarios(
            metric_series,
            growth_context=growth_context,
            statistical_context=statistical_context,
            multivariate_context=multivariate_context,
            preferred_metric_candidates=revenue_candidates,
        )
        payload = {
            "statistical_context": statistical_context,
            "multivariate_context": multivariate_context,
            "causal_inference": causal_context,
            "scenario_simulation": scenario_simulation,
        }
        dataset_confidence = dataset_confidence_from_state(state)
        primary_metric = str(growth_context.get("primary_metric") or "").strip()
        if primary_metric not in metric_series and metric_series:
            primary_metric = sorted(
                metric_series,
                key=lambda name: (-len(metric_series[name]), name),
            )[0]
        primary_values = metric_series.get(primary_metric, []) if primary_metric else []
        primary_horizons = growth_context.get("primary_horizons")
        if not isinstance(primary_horizons, dict):
            primary_horizons = {}
        anomaly_summary = statistical_context.get("anomaly_summary")
        if not isinstance(anomaly_summary, dict):
            anomaly_summary = {}
        correlation = multivariate_context.get("correlation")
        if not isinstance(correlation, dict):
            correlation = {}
        total_pairs = float(correlation.get("total_pairs") or 0.0)
        significant_pairs = float(correlation.get("significant_pairs") or 0.0)
        significant_pair_ratio = (
            (significant_pairs / total_pairs)
            if total_pairs > 0.0
            else 0.0
        )
        causal_summary = causal_context.get("summary")
        if not isinstance(causal_summary, dict):
            causal_summary = {}
        causal_pairs = float(causal_summary.get("pairs_tested") or 0.0)
        causal_significant = float(causal_summary.get("significant_edges") or 0.0)
        causal_ratio = (causal_significant / causal_pairs) if causal_pairs > 0.0 else 0.0
        scenarios = scenario_simulation.get("scenarios")
        if not isinstance(scenarios, dict):
            scenarios = {}
        worst = scenarios.get("worst") if isinstance(scenarios.get("worst"), dict) else {}
        best = scenarios.get("best") if isinstance(scenarios.get("best"), dict) else {}

        confidence_model = compute_standard_confidence(
            values=primary_values,
            signals={
                "short_growth": primary_horizons.get("short_growth"),
                "mid_growth": primary_horizons.get("mid_growth"),
                "long_growth": primary_horizons.get("long_growth"),
                "trend_acceleration": primary_horizons.get("trend_acceleration"),
                "anomaly_points": (
                    -float(anomaly_summary.get("total_anomaly_points"))
                    if isinstance(anomaly_summary.get("total_anomaly_points"), (int, float))
                    else 0.0
                ),
                "significant_pair_ratio": significant_pair_ratio,
                "causal_significant_ratio": causal_ratio,
                "worst_growth": worst.get("projected_growth"),
                "best_growth": best.get("projected_growth"),
            },
            dataset_confidence=dataset_confidence,
            upstream_confidences=[
                float(growth_context.get("confidence_score") or 0.0),
                float(statistical_context.get("confidence_score") or 0.0),
                float(multivariate_context.get("confidence_score") or 0.0),
                float(scenario_simulation.get("base_confidence") or 0.0),
            ],
            status=(
                "partial"
                if any(
                    str(src.get("status") or "").strip().lower() == "partial"
                    for src in (growth_context, statistical_context, multivariate_context, scenario_simulation)
                    if isinstance(src, dict)
                )
                else "success"
            ),
        )
        warnings: list[str] = []
        warnings.extend(str(item) for item in statistical_context.get("warnings", []))
        warnings.extend(str(item) for item in multivariate_context.get("warnings", []))
        warnings.extend(str(item) for item in causal_context.get("warnings", []))
        warnings.extend(str(item) for item in scenario_simulation.get("warnings", []))
        warnings.extend(str(item) for item in confidence_model.get("warnings", []))
        payload["confidence_breakdown"] = confidence_model
        return {
            "multivariate_scenario_data": success(
                payload,
                warnings=warnings,
                confidence_score=float(confidence_model["confidence_score"]),
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "multivariate_scenario_data": failed(
                str(exc),
                {"stage": "multivariate_scenario"},
            ),
        }
