"""
agent/nodes/category_formula_node.py

Run category-pack deterministic formula using registry bindings.
"""

from __future__ import annotations

from typing import Any, Mapping

from agent.helpers.confidence_model import compute_standard_confidence
from agent.helpers.kpi_extraction import (
    coerce_numeric,
    dataset_confidence_from_state,
    metric_series_from_kpi_payload,
    records_from_kpi_payload,
    resolve_kpi_payload,
)
from agent.nodes.node_result import failed, skipped, success
from agent.state import AgentState
from app.services.category_registry import CategoryRegistryError, require_category_pack


def _dependency_value(
    *,
    source: str,
    agg: Mapping[str, Any],
    extra: Mapping[str, Any],
    default: Any,
) -> Any:
    """Resolve a formula input binding (``agg.key`` or ``extra.key``)."""
    prefix, _, key = source.partition(".")
    prefix = prefix.strip().lower()
    key = key.strip()
    if prefix == "agg":
        value = agg.get(key)
        return default if value is None else value
    if prefix == "extra":
        return extra.get(key, default)
    return default


def _is_missing(value: Any, *, missing_when: str) -> bool:
    """Check if a dependency value is considered missing."""
    if missing_when == "is_empty":
        if value is None:
            return True
        if isinstance(value, (str, bytes)):
            return not value
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0
        return False
    return value is None


def category_formula_node(state: AgentState) -> AgentState:
    """Run category-pack deterministic formula using registry bindings."""
    try:
        business_type = str(state.get("business_type") or "").strip().lower()
        try:
            pack = require_category_pack(business_type)
        except CategoryRegistryError:
            pack = require_category_pack("general_timeseries")

        kpi_payload = resolve_kpi_payload(state)
        records = records_from_kpi_payload(kpi_payload)
        metric_series = metric_series_from_kpi_payload(kpi_payload)
        if not metric_series:
            return {
                "category_formula_data": skipped(
                    "kpi_unavailable",
                    {"category": pack.name, "records": len(records)},
                ),
            }
        aliases = pack.metric_aliases
        revenue_aliases = aliases.get("recurring_revenue", ("recurring_revenue",))
        active_aliases = aliases.get("active_customer_count", ("active_customer_count",))
        churn_aliases = aliases.get("churned_customer_count", ("churned_customer_count",))

        def _pick_series(candidates: tuple[str, ...]) -> list[float]:
            for candidate in candidates:
                if candidate in metric_series:
                    return list(metric_series[candidate])
            return []

        revenue_series = _pick_series(revenue_aliases)
        active_series = _pick_series(active_aliases)
        churn_series = _pick_series(churn_aliases)
        agg_values: dict[str, Any] = {
            "subscription_revenues": list(revenue_series),
            "active_customers": int(round(active_series[-1])) if active_series else 0,
            "lost_customers": int(round(churn_series[-1])) if churn_series else 0,
            "previous_revenue": float(revenue_series[-2]) if len(revenue_series) >= 2 else 0.0,
        }
        extra_values: dict[str, Any] = {}

        formula_inputs: dict[str, Any] = {}
        for key, binding in pack.formula_input_bindings.items():
            formula_inputs[key] = _dependency_value(
                source=binding.source,
                agg=agg_values,
                extra=extra_values,
                default=binding.default,
            )
        metrics = pack.formula.calculate(formula_inputs)

        optional_missing: list[str] = []
        required_missing: list[str] = []
        for metric_name, dependencies in pack.validity_rules.items():
            missing = []
            for dependency in dependencies:
                value = _dependency_value(
                    source=dependency.source,
                    agg=agg_values,
                    extra=extra_values,
                    default=None,
                )
                if _is_missing(value, missing_when=dependency.missing_when):
                    missing.append(dependency.source)
            if missing:
                if metric_name in pack.optional_signals:
                    optional_missing.append(metric_name)
                else:
                    required_missing.append(metric_name)

        metric_payload: dict[str, Any] = {}
        for name, value in metrics.items():
            metric_payload[name] = {
                "value": coerce_numeric(value),
                "status": (
                    "missing_optional"
                    if name in optional_missing
                    else ("missing_required" if name in required_missing else "success")
                ),
            }

        warnings: list[str] = []
        if optional_missing:
            warnings.append(f"Optional metrics unavailable: {sorted(optional_missing)}")
        if required_missing:
            warnings.append(f"Required formula dependencies missing: {sorted(required_missing)}")

        dataset_confidence = dataset_confidence_from_state(state)
        optional_ratio = (
            (len(optional_missing) / max(1, len(pack.optional_signals)))
            if pack.optional_signals
            else 0.0
        )
        confidence_signals: dict[str, Any] = {
            "optional_missing_ratio": -optional_ratio,
            "required_missing_count": -float(len(required_missing)),
        }
        for metric_name, metric_info in metric_payload.items():
            if not isinstance(metric_info, Mapping):
                continue
            if metric_info.get("status") == "success":
                confidence_signals[f"metric_{metric_name}"] = metric_info.get("value")

        confidence_model = compute_standard_confidence(
            values=revenue_series,
            signals=confidence_signals,
            dataset_confidence=dataset_confidence,
            upstream_confidences=[round(1.0 - (optional_ratio * 0.4), 6)],
            status="partial" if (optional_missing or required_missing) else "success",
            base_warnings=warnings,
        )
        payload = {
            "category": pack.name,
            "required_fields": list(pack.required_inputs),
            "canonical_metric_aliases": {k: list(v) for k, v in pack.metric_aliases.items()},
            "optional_missing": sorted(optional_missing),
            "metrics": metric_payload,
            "confidence_breakdown": confidence_model,
        }
        return {
            "category_formula_data": success(
                payload,
                warnings=confidence_model["warnings"],
                confidence_score=float(confidence_model["confidence_score"]),
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "category_formula_data": failed(str(exc), {"stage": "category_formula"}),
        }
