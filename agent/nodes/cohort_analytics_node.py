"""
agent/nodes/cohort_analytics_node.py

Compute cohort analytics when cohort keys are available.

When no cohort rows exist, synthesizes retention estimates from
aggregate churn data using exponential decay approximation.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.helpers.canonical_queries import cohort_rows_for_records
from agent.helpers.confidence_model import compute_standard_confidence
from agent.helpers.kpi_extraction import (
    dataset_confidence_from_state,
    extract_numeric_metric,
    metric_series_from_kpi_payload,
    records_from_kpi_payload,
    resolve_kpi_payload,
    resolve_period_bounds,
)
from agent.nodes.node_result import failed, skipped, success
from agent.state import AgentState
from app.services.cohort_analytics import DEFAULT_COHORT_KEYS, compute_cohort_analytics
from app.services.kpi_canonical_schema import metric_aliases_for_business_type
from app.services.statistics.survival_analysis import survival_from_retention_curve

logger = logging.getLogger(__name__)

# Confidence penalty applied when using synthetic cohort estimation.
_SYNTHETIC_COHORT_PENALTY = -0.1

# Number of periods to project retention decay forward.
_SYNTHETIC_RETENTION_PERIODS = 12

# Churn metric candidates across business types.
_CHURN_METRIC_CANDIDATES = (
    "churn_rate",
    "client_churn",
    "customer_churn_rate",
    "monthly_churn_rate",
    "churn",
)


def _estimate_synthetic_cohort(
    churn_rate: float,
    periods: int = _SYNTHETIC_RETENTION_PERIODS,
) -> dict[str, Any]:
    """Estimate cohort retention from aggregate churn rate.

    Uses the approximation: retention_t = retention_(t-1) * (1 - churn_rate)

    Args:
        churn_rate: Period-over-period churn rate (0.0 to 1.0).
        periods: Number of periods to project forward.

    Returns:
        Synthetic cohort payload with retention_decay, churn_acceleration,
        and lifetime_estimate.
    """
    # Normalize churn_rate to [0, 1]
    if churn_rate > 1.0:
        churn_rate = churn_rate / 100.0
    churn_rate = max(0.0, min(1.0, churn_rate))

    if churn_rate <= 0.0:
        return {
            "method": "synthetic_cohort_estimation",
            "status": "insufficient_data",
            "reason": "churn_rate is zero or negative; cannot estimate retention decay",
            "retention_decay": 0.0,
            "churn_acceleration": 0.0,
            "lifetime_estimate": None,
            "retention_curve": [1.0] * periods,
        }

    # Build retention curve: retention_t = retention_(t-1) * (1 - churn_rate)
    retention_curve: list[float] = []
    retention = 1.0
    for _ in range(periods):
        retention_curve.append(round(retention, 6))
        retention *= (1.0 - churn_rate)

    # Retention decay: difference between first and last period
    retention_decay = round(retention_curve[0] - retention_curve[-1], 6)

    # Churn acceleration: second derivative approximation
    if len(retention_curve) >= 3:
        deltas = [
            retention_curve[i] - retention_curve[i - 1]
            for i in range(1, len(retention_curve))
        ]
        accelerations = [deltas[i] - deltas[i - 1] for i in range(1, len(deltas))]
        churn_acceleration = round(
            sum(accelerations) / len(accelerations), 6,
        ) if accelerations else 0.0
    else:
        churn_acceleration = 0.0

    # Lifetime estimate: 1 / churn_rate (expected periods until churn)
    lifetime_estimate = round(1.0 / churn_rate, 2)

    return {
        "method": "synthetic_cohort_estimation",
        "status": "success",
        "churn_rate_used": round(churn_rate, 6),
        "retention_decay": retention_decay,
        "churn_acceleration": churn_acceleration,
        "lifetime_estimate": lifetime_estimate,
        "retention_curve": retention_curve,
        "periods_projected": periods,
    }


_REVENUE_METRIC_CANDIDATES = (
    "revenue",
    "mrr",
    "total_revenue",
    "net_revenue",
    "sales",
)


def _estimate_implied_churn_from_revenue(
    metric_series: dict[str, list[float]],
) -> float | None:
    """Estimate implied monthly churn rate from revenue decline patterns.

    When no explicit churn metric exists, sustained revenue decline implies
    customer attrition.  This uses the average negative period-over-period
    change as a proxy.

    Returns a churn rate in [0, 1] or None if revenue is not declining.
    """
    for candidate in _REVENUE_METRIC_CANDIDATES:
        series = metric_series.get(candidate)
        if not series or len(series) < 4:
            continue

        # Compute period-over-period rates of change
        changes: list[float] = []
        for i in range(1, len(series)):
            prev = series[i - 1]
            if abs(prev) < 1e-9:
                continue
            changes.append((series[i] - prev) / abs(prev))

        if not changes:
            continue

        negative_changes = [c for c in changes if c < 0]
        if len(negative_changes) < len(changes) * 0.4:
            # Revenue is not predominantly declining — no implied churn
            continue

        # Use mean negative change magnitude as implied churn proxy,
        # scaled conservatively (revenue loss != customer loss 1:1).
        avg_decline = abs(sum(negative_changes) / len(negative_changes))
        implied_churn = min(0.5, avg_decline * 0.7)  # conservative scale
        if implied_churn >= 0.01:
            return round(implied_churn, 6)

    return None


def _extract_churn_rate(state: AgentState) -> tuple[float | None, str]:
    """Extract the most recent churn rate from KPI payload.

    Returns (churn_rate, source) where source is one of:
    'explicit', 'revenue_implied', or 'none'.
    """
    kpi_payload = resolve_kpi_payload(state)
    if not kpi_payload:
        return None, "none"

    # Try from computed KPIs in records (most recent period)
    records = records_from_kpi_payload(kpi_payload)
    if records:
        # Sort by period_end descending to get most recent
        sorted_records = sorted(
            records,
            key=lambda r: str(r.get("period_end") or ""),
            reverse=True,
        )
        for record in sorted_records:
            computed = record.get("computed_kpis")
            if not isinstance(computed, dict):
                continue
            value = extract_numeric_metric(computed, _CHURN_METRIC_CANDIDATES)
            if value is not None and value > 0:
                return value, "explicit"

    # Try from metric series (use last value)
    metric_series = metric_series_from_kpi_payload(kpi_payload)
    for candidate in _CHURN_METRIC_CANDIDATES:
        series = metric_series.get(candidate)
        if series and len(series) > 0:
            last_value = series[-1]
            if last_value > 0:
                return last_value, "explicit"

    # Fallback: estimate implied churn from revenue decline
    implied = _estimate_implied_churn_from_revenue(metric_series)
    if implied is not None:
        return implied, "revenue_implied"

    return None, "none"


def cohort_analytics_node(state: AgentState) -> AgentState:
    """Compute cohort analytics when cohort keys are available.

    When no cohort rows exist, falls back to synthetic retention
    estimation from aggregate churn data.
    """
    try:
        kpi_payload = resolve_kpi_payload(state)
        records = records_from_kpi_payload(kpi_payload)
        if not isinstance(kpi_payload, dict):
            return {"cohort_data": skipped("kpi_unavailable", {"records": 0})}

        business_type = str(state.get("business_type") or "").strip().lower()
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

        # ── Fallback: synthetic cohort estimation from churn data ──
        if not cohort_rows:
            churn_rate, churn_source = _extract_churn_rate(state)
            if churn_rate is None or churn_rate <= 0:
                return {
                    "cohort_data": skipped(
                        "cohort_not_applicable",
                        {
                            "records": len(records),
                            "record_count": int(kpi_payload.get("record_count") or 0),
                        },
                    ),
                }

            logger.info(
                "No cohort rows for %s; computing synthetic cohort from "
                "churn_rate=%.4f (source=%s)",
                entity_name,
                churn_rate,
                churn_source,
            )
            synthetic = _estimate_synthetic_cohort(churn_rate)
            synthetic["churn_source"] = churn_source

            dataset_confidence = dataset_confidence_from_state(state)
            # Revenue-implied churn is a weaker signal than explicit churn
            method_penalty = (
                -0.8 if churn_source == "revenue_implied" else -0.5
            )
            confidence_model = compute_standard_confidence(
                values=synthetic.get("retention_curve", []),
                signals={
                    "retention_decay": -synthetic.get("retention_decay", 0.0),
                    "lifetime_estimate": synthetic.get("lifetime_estimate"),
                    "churn_acceleration": -abs(synthetic.get("churn_acceleration", 0.0)),
                    "synthetic_method": method_penalty,
                },
                dataset_confidence=dataset_confidence,
                upstream_confidences=[],
                status="success" if synthetic.get("status") == "success" else "insufficient_data",
                base_warnings=[
                    f"Cohort estimated synthetically from {churn_source} churn data; "
                    "confidence penalty applied.",
                ],
            )

            # Apply synthetic cohort penalty
            raw_conf = float(confidence_model["confidence_score"])
            confidence_model["confidence_score"] = round(
                max(0.0, raw_conf + _SYNTHETIC_COHORT_PENALTY), 6,
            )
            confidence_model["synthetic_penalty"] = _SYNTHETIC_COHORT_PENALTY

            cohort_payload = {
                **synthetic,
                "cohort_count": 0,
                "entity_name": entity_name,
                "confidence_breakdown": confidence_model,
                "signals": {
                    "retention_decay": synthetic.get("retention_decay"),
                    "churn_acceleration": synthetic.get("churn_acceleration"),
                    "lifetime_estimate": synthetic.get("lifetime_estimate"),
                    "risk_hint": (
                        "high" if synthetic.get("retention_decay", 0) > 0.5
                        else "moderate" if synthetic.get("retention_decay", 0) > 0.25
                        else "low"
                    ),
                },
            }
            return {
                "cohort_data": success(
                    cohort_payload,
                    warnings=confidence_model["warnings"],
                    confidence_score=float(confidence_model["confidence_score"]),
                ),
            }

        # ── Standard cohort analytics path ──
        aliases = metric_aliases_for_business_type(business_type)
        active_candidates = aliases.get("active_customer_count", ("active_customer_count",))
        churn_candidates = aliases.get("churned_customer_count", ("churned_customer_count",))
        cohort = compute_cohort_analytics(
            cohort_rows,
            cohort_keys=DEFAULT_COHORT_KEYS,
            active_metric_names=active_candidates,
            churn_metric_names=churn_candidates,
        )

        dataset_confidence = dataset_confidence_from_state(state)
        warnings = [str(item) for item in cohort.get("warnings", [])]
        signals = cohort.get("signals")
        if not isinstance(signals, dict):
            signals = {}
        retention_values: list[float] = []
        survival_profiles: list[dict[str, object]] = []
        survival_warnings: list[str] = []
        survival_medians: list[float] = []
        survival_means: list[float] = []
        cohorts_by_key = cohort.get("cohorts_by_key")
        if isinstance(cohorts_by_key, dict):
            for cohort_key, key_payload in cohorts_by_key.items():
                if not isinstance(key_payload, dict):
                    continue
                cohorts = key_payload.get("cohorts")
                if not isinstance(cohorts, list):
                    continue
                for cohort_item in cohorts:
                    if not isinstance(cohort_item, dict):
                        continue
                    retention_curve = cohort_item.get("retention_curve")
                    if not isinstance(retention_curve, list):
                        continue
                    for point in retention_curve:
                        if not isinstance(point, dict):
                            continue
                        value = point.get("retention_rate")
                        if isinstance(value, (int, float)):
                            retention_values.append(float(value))
                    if retention_curve:
                        rates = [
                            float(point.get("retention_rate"))
                            for point in retention_curve
                            if isinstance(point, dict)
                            and isinstance(point.get("retention_rate"), (int, float))
                        ]
                        total_customers = None
                        first_point = retention_curve[0] if retention_curve else {}
                        if isinstance(first_point, dict):
                            initial_active = first_point.get("active_customers")
                            if isinstance(initial_active, (int, float)):
                                total_customers = max(1, int(initial_active))
                        survival_result = survival_from_retention_curve(
                            rates,
                            total_customers=total_customers,
                        )
                        if isinstance(survival_result, dict):
                            survival_profiles.append(
                                {
                                    "cohort_key": str(cohort_key),
                                    "cohort_value": str(cohort_item.get("cohort_value") or ""),
                                    "status": str(survival_result.get("status") or "unknown"),
                                    "median_survival": survival_result.get("median_survival"),
                                    "mean_survival": survival_result.get("mean_survival"),
                                    "hazard_rates": survival_result.get("hazard_rates", []),
                                    "warnings": survival_result.get("warnings", []),
                                }
                            )
                            median_value = survival_result.get("median_survival")
                            mean_value = survival_result.get("mean_survival")
                            if isinstance(median_value, (int, float)):
                                survival_medians.append(float(median_value))
                            if isinstance(mean_value, (int, float)):
                                survival_means.append(float(mean_value))
                            survival_warnings.extend(
                                str(item)
                                for item in survival_result.get("warnings", [])
                                if str(item).strip()
                            )

        survival_summary = {
            "status": "success" if survival_profiles else "insufficient_data",
            "profiles_count": len(survival_profiles),
            "profiles": survival_profiles,
            "median_survival_estimate": (
                round(sum(survival_medians) / len(survival_medians), 6)
                if survival_medians
                else None
            ),
            "mean_survival_estimate": (
                round(sum(survival_means) / len(survival_means), 6)
                if survival_means
                else None
            ),
            "warnings": sorted(set(survival_warnings)),
        }

        confidence_model = compute_standard_confidence(
            values=retention_values,
            signals={
                "retention_decay": (
                    -float(signals.get("retention_decay"))
                    if isinstance(signals.get("retention_decay"), (int, float))
                    else None
                ),
                "lifetime_estimate": signals.get("lifetime_estimate"),
                "churn_acceleration": (
                    -float(signals.get("churn_acceleration"))
                    if isinstance(signals.get("churn_acceleration"), (int, float))
                    else None
                ),
                "risk_hint": (
                    -1.0
                    if str(signals.get("risk_hint") or "").strip().lower() == "high"
                    else (
                        -0.5
                        if str(signals.get("risk_hint") or "").strip().lower() == "moderate"
                        else 1.0
                    )
                ),
                "sparse_cohorts": (
                    -float(signals.get("sparse_cohorts"))
                    if isinstance(signals.get("sparse_cohorts"), (int, float))
                    else 0.0
                ),
                "survival_median": survival_summary.get("median_survival_estimate"),
                "survival_mean": survival_summary.get("mean_survival_estimate"),
            },
            dataset_confidence=dataset_confidence,
            upstream_confidences=[float(cohort.get("confidence_score") or 0.0)],
            status=str(cohort.get("status") or "success"),
            base_warnings=warnings + survival_summary["warnings"],
        )
        cohort_payload = {
            **cohort,
            "survival_analysis": survival_summary,
            "confidence_breakdown": confidence_model,
        }
        return {
            "cohort_data": success(
                cohort_payload,
                warnings=confidence_model["warnings"],
                confidence_score=float(confidence_model["confidence_score"]),
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"cohort_data": failed(str(exc), {"stage": "cohort_analytics"})}
