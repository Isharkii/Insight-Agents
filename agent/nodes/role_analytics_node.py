"""
agent/nodes/role_analytics_node.py

Role analytics stage: KPI -> Role Aggregation -> Role Scoring.
Also builds deterministic competitive context from canonical benchmark data.

This node reads upstream payloads (growth, cohort, timeseries_factors, etc.)
but does NOT re-compute them on failure.  If an upstream node failed, its
envelope status is propagated as a warning — not silently masked.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from agent.helpers.canonical_queries import (
    fetch_canonical_dimension_rows,
    fetch_macro_context_rows,
)
from agent.helpers.confidence_model import compute_standard_confidence
from agent.helpers.kpi_extraction import (
    dataset_confidence_from_state,
    metric_series_from_kpi_payload,
    records_from_kpi_payload,
    resolve_kpi_payload,
    resolve_period_bounds,
    rows_from_kpi_payload,
)
from agent.nodes.node_result import (
    confidence_of,
    failed,
    payload_of,
    skipped,
    status_of,
    success,
)
from agent.state import AgentState
from app.services.kpi_canonical_schema import metric_aliases_for_business_type
from app.services.macro_context_service import build_macro_context
from app.services.role_dimension_analytics import build_role_dimension_summary
from app.services.role_performance_scoring import score_role_performance


def _no_competitive_context(warnings: list[str] | None = None) -> dict[str, Any]:
    """Default competitive context when no peers are available."""
    return {
        "available": False,
        "source": "unavailable",
        "peer_count": 0,
        "peers": [],
        "metrics": [],
        "benchmark_rows_count": 0,
        "numeric_signals": [],
        "cache_hit": False,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "warnings": warnings or [],
    }


# Upstream node keys whose payloads are bundled into segmentation output
# and whose warnings/confidence propagate into role analytics.
_UPSTREAM_KEYS = (
    "growth_data",
    "timeseries_factors_data",
    "cohort_data",
    "category_formula_data",
    "unit_economics_data",
    "multivariate_scenario_data",
)


def _read_upstream_payload(state: AgentState, key: str) -> dict[str, Any]:
    """Read an upstream payload — return empty dict if unavailable, never re-compute."""
    return payload_of(state.get(key)) or {}


def role_analytics_node(state: AgentState) -> AgentState:
    """Role analytics + deterministic competitive context builder."""
    try:
        kpi_payload = resolve_kpi_payload(state)
        records = records_from_kpi_payload(kpi_payload)
        if not isinstance(kpi_payload, dict):
            return {
                "segmentation": skipped("kpi_unavailable", {"records": 0}),
            }

        business_type = str(state.get("business_type") or "").strip().lower()
        entity_name = str(
            kpi_payload.get("fetched_for")
            or state.get("entity_name")
            or ""
        ).strip()

        period_start, period_end = resolve_period_bounds(kpi_payload)
        canonical_rows: list[dict[str, Any]] = []
        if entity_name:
            canonical_rows = fetch_canonical_dimension_rows(
                entity_name=entity_name,
                business_type=business_type,
                period_start=period_start,
                period_end=period_end,
            )

        empty_ctx = _no_competitive_context()

        if not canonical_rows:
            canonical_rows = rows_from_kpi_payload(records)
        if not canonical_rows:
            return {
                "segmentation": skipped("dimension_values_missing", {"records": len(records)}),
                "competitive_context": empty_ctx,
            }

        summary = build_role_dimension_summary(canonical_rows, top_n=3)
        if not summary.get("records_used"):
            return {
                "segmentation": skipped("dimension_values_missing", {"records": len(canonical_rows)}),
                "competitive_context": empty_ctx,
            }

        # ── Macro context & competitive context (DB queries) ──────────
        inflation_rows: list[dict[str, Any]] = []
        benchmark_rows: list[dict[str, Any]] = []
        if entity_name:
            try:
                inflation_rows, benchmark_rows = fetch_macro_context_rows(
                    entity_name=entity_name,
                    business_type=business_type,
                    period_start=period_start,
                    period_end=period_end,
                )
            except Exception:
                inflation_rows = []
                benchmark_rows = []

        peer_entities: set[str] = set()
        benchmark_metrics: set[str] = set()
        for row in benchmark_rows:
            ename = str(row.get("entity_name") or "").strip()
            mname = str(row.get("metric_name") or "").strip()
            if ename:
                peer_entities.add(ename)
            if mname:
                benchmark_metrics.add(mname)

        _has_peers = len(peer_entities) > 0
        competitive_context: dict[str, Any] = {
            "available": _has_peers,
            "source": "deterministic_local" if _has_peers else "unavailable",
            "peer_count": len(peer_entities),
            "peers": sorted(peer_entities),
            "metrics": sorted(benchmark_metrics),
            "benchmark_rows_count": len(benchmark_rows),
            "numeric_signals": [],
            "cache_hit": False,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "warnings": [],
        }

        aliases = metric_aliases_for_business_type(business_type)
        revenue_candidates = aliases.get("recurring_revenue", ("recurring_revenue",))
        metric_series = metric_series_from_kpi_payload(kpi_payload)

        macro_context = build_macro_context(
            kpi_payload=kpi_payload,
            inflation_rows=inflation_rows,
            benchmark_rows=benchmark_rows,
            metric_candidates=revenue_candidates,
        )

        # ── Read upstream payloads (no fallback re-computation) ───────
        growth_context = _read_upstream_payload(state, "growth_data")
        timeseries_factors = _read_upstream_payload(state, "timeseries_factors_data")
        cohort_analytics = _read_upstream_payload(state, "cohort_data")
        category_formula = _read_upstream_payload(state, "category_formula_data")
        unit_economics = _read_upstream_payload(state, "unit_economics_data")
        multivariate_bundle = _read_upstream_payload(state, "multivariate_scenario_data")

        statistical_context = multivariate_bundle.get("statistical_context")
        if not isinstance(statistical_context, Mapping):
            statistical_context = {}
        multivariate_context = multivariate_bundle.get("multivariate_context")
        if not isinstance(multivariate_context, Mapping):
            multivariate_context = {}
        scenario_simulation = multivariate_bundle.get("scenario_simulation")
        if not isinstance(scenario_simulation, Mapping):
            scenario_simulation = {}

        # ── Role dimension scoring ────────────────────────────────────
        scoring_input: dict[str, dict[str, Any]] = {}
        for dimension, dim_payload in summary.get("by_dimension", {}).items():
            contributors = dim_payload.get("contributors", [])
            if not isinstance(contributors, list):
                continue
            for contributor in contributors:
                if not isinstance(contributor, Mapping):
                    continue
                name = str(contributor.get("name") or "").strip()
                if not name:
                    continue
                share = float(contributor.get("contribution_share") or 0.0)
                key = f"{dimension}:{name}"
                scoring_input[key] = {
                    "growth_rate": [share],
                    "efficiency_metric": [share],
                    "contribution_weight": [share],
                    "stability_series": [share],
                }

        role_scores = score_role_performance(scoring_input, category=business_type)

        # ── Upstream diagnostics ──────────────────────────────────────
        node_warnings: list[str] = []
        upstream_confidences: list[float] = []
        has_partial_upstream = False
        for key in _UPSTREAM_KEYS:
            envelope = state.get(key)
            if isinstance(envelope, Mapping):
                node_warnings.extend(
                    str(item)
                    for item in envelope.get("warnings", [])
                    if str(item).strip()
                )
                upstream_confidences.append(confidence_of(envelope))
                if status_of(envelope) in {"insufficient_data", "skipped", "failed"}:
                    has_partial_upstream = True
            payload = payload_of(envelope)
            if isinstance(payload, Mapping):
                payload_status = str(payload.get("status") or "").strip().lower()
                if payload_status in {"partial", "insufficient_data"}:
                    has_partial_upstream = True

        # ── Confidence model ──────────────────────────────────────────
        primary_metric = str(growth_context.get("primary_metric") or "").strip()
        if primary_metric not in metric_series and metric_series:
            primary_metric = sorted(
                metric_series,
                key=lambda name: (-len(metric_series[name]), name),
            )[0]
        primary_values = metric_series.get(primary_metric, []) if primary_metric else []
        growth_horizons = growth_context.get("primary_horizons")
        if not isinstance(growth_horizons, Mapping):
            growth_horizons = {}
        cohort_signals = cohort_analytics.get("signals")
        if not isinstance(cohort_signals, Mapping):
            cohort_signals = {}
        scenario_scenarios = scenario_simulation.get("scenarios") if isinstance(scenario_simulation, Mapping) else {}
        if not isinstance(scenario_scenarios, Mapping):
            scenario_scenarios = {}
        scenario_worst = scenario_scenarios.get("worst")
        if not isinstance(scenario_worst, Mapping):
            scenario_worst = {}
        concentration = summary.get("dependency_concentration")
        if not isinstance(concentration, Mapping):
            concentration = {}
        most_concentrated = concentration.get("most_concentrated_dimension")
        if not isinstance(most_concentrated, Mapping):
            most_concentrated = {}

        confidence_model = compute_standard_confidence(
            values=primary_values,
            signals={
                "growth_short": growth_horizons.get("short_growth"),
                "growth_mid": growth_horizons.get("mid_growth"),
                "growth_long": growth_horizons.get("long_growth"),
                "growth_trend_acceleration": growth_horizons.get("trend_acceleration"),
                "cohort_churn_acceleration": (
                    -float(cohort_signals.get("churn_acceleration"))
                    if isinstance(cohort_signals.get("churn_acceleration"), (int, float))
                    else None
                ),
                "scenario_worst_growth": scenario_worst.get("projected_growth"),
                "dependency_hhi": (
                    -float(most_concentrated.get("hhi"))
                    if isinstance(most_concentrated.get("hhi"), (int, float))
                    else None
                ),
                "records_used": summary.get("records_used"),
            },
            dataset_confidence=dataset_confidence_from_state(state),
            upstream_confidences=upstream_confidences,
            status="partial" if has_partial_upstream else "success",
            base_warnings=node_warnings,
        )

        role_analytics_data = success(
            {
                "dimensions": summary.get("dimensions", []),
                "top_contributors": summary.get("top_contributors", []),
                "laggards": summary.get("laggards", []),
                "dependency_concentration": summary.get("dependency_concentration", {}),
                "by_dimension": summary.get("by_dimension", {}),
                "records_scanned": summary.get("records_scanned", 0),
                "records_used": summary.get("records_used", 0),
                "role_scoring": role_scores,
                "macro_context": macro_context,
                "cohort_analytics": cohort_analytics,
                "timeseries_factors": timeseries_factors,
                "category_formula": category_formula,
                "unit_economics": unit_economics,
                "statistical_context": statistical_context,
                "growth_context": growth_context,
                "multivariate_context": multivariate_context,
                "scenario_simulation": scenario_simulation,
                "confidence_breakdown": confidence_model,
            },
            warnings=confidence_model["warnings"],
            confidence_score=float(confidence_model["confidence_score"]),
        )
        return {
            "segmentation": role_analytics_data,
            "competitive_context": competitive_context,
        }

    except Exception as exc:  # noqa: BLE001
        return {
            "segmentation": failed(str(exc), {"stage": "role_analytics_node"}),
            "competitive_context": _no_competitive_context([f"segmentation_failed: {exc}"]),
        }
