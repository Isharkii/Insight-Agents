"""
Unified unit economics analysis for deterministic decision intelligence.

This module computes core unit economics metrics from KPI payload records:
    - LTV
    - CAC
    - LTV/CAC ratio
    - churn rate
    - revenue per customer

It also generates health signals such as:
    - healthy_growth
    - unsustainable_growth
    - acquisition_inefficiency
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

_ZERO_GUARD = 1e-9


@dataclass(frozen=True)
class UnitEconomicsConfig:
    """Threshold configuration for unit economics signal generation."""

    # LTV/CAC thresholds
    ltv_cac_healthy: float = 3.0
    ltv_cac_marginal: float = 1.0
    ltv_cac_critical: float = 1.0

    # Churn thresholds
    churn_healthy: float = 0.05
    churn_warning: float = 0.10
    churn_critical: float = 0.15

    # Growth and trend thresholds
    revenue_growth_healthy: float = 0.02
    cac_increase_warning: float = 0.10


_METRIC_ALIASES_BY_TYPE: dict[str, dict[str, tuple[str, ...]]] = {
    "saas": {
        "ltv": ("ltv",),
        "cac": ("cac", "customer_acquisition_cost", "blended_cac"),
        "churn_rate": ("churn_rate",),
        "revenue_per_customer": ("arpu", "revenue_per_customer"),
        "revenue": ("mrr", "recurring_revenue"),
        "growth_rate": ("growth_rate",),
    },
    "ecommerce": {
        "ltv": ("ltv",),
        "cac": ("cac", "customer_acquisition_cost"),
        "churn_rate": ("churn_rate", "customer_churn"),
        "revenue_per_customer": ("aov", "revenue_per_customer"),
        "revenue": ("revenue", "gmv", "recurring_revenue"),
        "growth_rate": ("growth_rate",),
    },
    "agency": {
        "ltv": ("client_ltv", "ltv"),
        "cac": ("cac", "customer_acquisition_cost"),
        "churn_rate": ("client_churn", "churn_rate"),
        "revenue_per_customer": ("revenue_per_employee", "revenue_per_customer"),
        "revenue": ("total_revenue", "retainer_revenue", "recurring_revenue"),
        "growth_rate": ("growth_rate",),
    },
}

_DEFAULT_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "ltv": ("ltv", "client_ltv"),
    "cac": ("cac", "customer_acquisition_cost"),
    "churn_rate": ("churn_rate", "client_churn"),
    "revenue_per_customer": ("arpu", "aov", "revenue_per_customer", "revenue_per_employee"),
    "revenue": ("recurring_revenue", "revenue", "mrr", "total_revenue"),
    "growth_rate": ("growth_rate",),
}


@dataclass(frozen=True)
class UnitEconomicsSignal:
    """Generated deterministic signal with severity and machine-readable evidence."""

    signal: str
    severity: str  # info | warning | critical
    confidence: float  # 0..1
    description: str
    evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "severity": self.severity,
            "confidence": round(self.confidence, 6),
            "description": self.description,
            "evidence": dict(self.evidence),
        }


def _aliases_for_business_type(business_type: str) -> dict[str, tuple[str, ...]]:
    normalized = str(business_type or "").strip().lower()
    return _METRIC_ALIASES_BY_TYPE.get(normalized, _DEFAULT_METRIC_ALIASES)


def _parse_iso_epoch(value: Any) -> float:
    if not isinstance(value, str) or not value.strip():
        return float("-inf")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return float("-inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _sort_records(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            _parse_iso_epoch(record.get("period_end")),
            _parse_iso_epoch(record.get("created_at")),
        ),
    )


def _to_float_metric(raw_value: Any) -> float | None:
    if isinstance(raw_value, Mapping):
        if raw_value.get("is_valid") is False:
            return None
        if raw_value.get("error") is not None:
            return None
        raw_value = raw_value.get("value")
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _extract_metric(computed_kpis: Mapping[str, Any], aliases: tuple[str, ...]) -> float | None:
    for name in aliases:
        if name not in computed_kpis:
            continue
        value = _to_float_metric(computed_kpis.get(name))
        if value is not None:
            return value
    return None


def _extract_series(records: Sequence[Mapping[str, Any]], aliases: tuple[str, ...]) -> list[float]:
    series: list[float] = []
    for record in records:
        computed = record.get("computed_kpis")
        if not isinstance(computed, Mapping):
            continue
        value = _extract_metric(computed, aliases)
        if value is not None:
            series.append(value)
    return series


def compute_ltv_cac_ratio(ltv: float | None, cac: float | None) -> float | None:
    if ltv is None or cac is None:
        return None
    if abs(cac) < _ZERO_GUARD:
        return None
    return ltv / cac


def compute_revenue_per_customer(revenue: float | None, customer_count: float | None) -> float | None:
    if revenue is None or customer_count is None:
        return None
    if customer_count < _ZERO_GUARD:
        return None
    return revenue / customer_count


def _pct_change(current: float, previous: float) -> float | None:
    if abs(previous) < _ZERO_GUARD:
        return None
    return (current - previous) / previous


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def _insufficient_result(reason: str) -> dict[str, Any]:
    return {
        "metrics": {
            "ltv": None,
            "cac": None,
            "ltv_cac_ratio": None,
            "churn_rate": None,
            "revenue_per_customer": None,
            "revenue": None,
            "growth_rate": None,
        },
        "metric_series": {},
        "trends": {},
        "signals": [
            {
                "signal": "data_insufficient",
                "severity": "warning",
                "confidence": 1.0,
                "description": reason,
                "evidence": {},
            }
        ],
        "signal_summary": "data_insufficient",
        "confidence": 0.0,
        "available_metrics": 0,
        "status": "insufficient_data",
        "warnings": [reason],
        "config": {},
    }


# ---------------------------------------------------------------------------
# Fallback: estimate unit economics from revenue trajectory
# ---------------------------------------------------------------------------

_REVENUE_CANDIDATES = (
    "revenue", "mrr", "recurring_revenue", "total_revenue", "net_revenue", "sales",
)


def _estimate_implied_churn_from_revenue(
    revenue_series: list[float],
) -> float | None:
    """Estimate implied monthly churn from sustained revenue decline.

    Uses average negative period-over-period change as a proxy.
    Returns churn rate in [0, 1] or None if revenue is not declining.
    """
    if len(revenue_series) < 4:
        return None

    changes: list[float] = []
    for i in range(1, len(revenue_series)):
        prev = revenue_series[i - 1]
        if abs(prev) < _ZERO_GUARD:
            continue
        changes.append((revenue_series[i] - prev) / abs(prev))

    if not changes:
        return None

    negative = [c for c in changes if c < 0]
    if len(negative) < len(changes) * 0.4:
        return None

    avg_decline = abs(sum(negative) / len(negative))
    implied = min(0.5, avg_decline * 0.7)
    return round(implied, 6) if implied >= 0.01 else None


def _compute_growth_efficiency(
    revenue_series: list[float],
    growth_series: list[float] | None,
) -> dict[str, Any]:
    """Compute growth efficiency metrics from revenue trajectory.

    Returns revenue_velocity, growth_stability, and burn_risk_indicator.
    """
    n = len(revenue_series)
    if n < 3:
        return {}

    # Revenue velocity: average period-over-period growth rate
    changes: list[float] = []
    for i in range(1, n):
        prev = revenue_series[i - 1]
        if abs(prev) < _ZERO_GUARD:
            continue
        changes.append((revenue_series[i] - prev) / abs(prev))

    if not changes:
        return {}

    avg_growth = sum(changes) / len(changes)

    # Growth stability: coefficient of variation of growth rates
    if len(changes) >= 2:
        mean_change = sum(changes) / len(changes)
        variance = sum((c - mean_change) ** 2 for c in changes) / len(changes)
        growth_cv = (variance ** 0.5) / max(abs(mean_change), _ZERO_GUARD)
    else:
        growth_cv = 0.0

    # Growth acceleration: compare recent half vs first half
    mid = len(changes) // 2
    if mid > 0:
        first_half = sum(changes[:mid]) / mid
        second_half = sum(changes[mid:]) / max(1, len(changes) - mid)
        acceleration = second_half - first_half
    else:
        acceleration = 0.0

    # Burn risk: declining revenue + high volatility
    declining_periods = sum(1 for c in changes if c < 0)
    decline_ratio = declining_periods / len(changes)

    if decline_ratio > 0.6 and growth_cv > 0.3:
        burn_risk = "high"
    elif decline_ratio > 0.4 or growth_cv > 0.5:
        burn_risk = "moderate"
    else:
        burn_risk = "low"

    return {
        "revenue_velocity": round(avg_growth, 6),
        "growth_stability_cv": round(growth_cv, 6),
        "growth_acceleration": round(acceleration, 6),
        "decline_ratio": round(decline_ratio, 6),
        "burn_risk_indicator": burn_risk,
        "periods_analyzed": len(changes),
    }


def _estimate_unit_economics_from_revenue(
    ordered_records: Sequence[Mapping[str, Any]],
    aliases: dict[str, tuple[str, ...]],
    metrics: dict[str, float | None],
    metric_series: dict[str, list[float]],
    trends: dict[str, float | None],
    has_trend_data: bool,
    warnings: list[str],
    cfg: UnitEconomicsConfig,
) -> dict[str, Any] | None:
    """Fallback: estimate unit economics from revenue trajectory alone.

    When explicit LTV/CAC/churn metrics are unavailable, derives:
    - implied_churn from revenue decline patterns
    - estimated_ltv from revenue / implied_churn
    - growth efficiency metrics
    - burn risk indicator

    Returns None if insufficient revenue data exists.
    """
    # Find best revenue series
    revenue_series: list[float] | None = None
    for candidate in _REVENUE_CANDIDATES:
        series = metric_series.get(candidate)
        if series and len(series) >= 3:
            revenue_series = series
            break

    # Try extracting from records directly
    if revenue_series is None:
        all_revenue_aliases = aliases.get("revenue", _DEFAULT_METRIC_ALIASES.get("revenue", ()))
        revenue_series = _extract_series(ordered_records, all_revenue_aliases)
        if len(revenue_series) < 3:
            return None

    revenue = metrics.get("revenue")
    growth = metrics.get("growth_rate")

    # Estimate implied churn from revenue decline
    implied_churn = _estimate_implied_churn_from_revenue(revenue_series)
    churn_source = "revenue_implied" if implied_churn is not None else "unavailable"

    # Estimate LTV from revenue and implied churn
    estimated_ltv: float | None = None
    if implied_churn is not None and revenue is not None and implied_churn > _ZERO_GUARD:
        estimated_ltv = round(revenue / implied_churn, 6)

    # Compute growth efficiency
    growth_series = metric_series.get("growth_rate")
    efficiency = _compute_growth_efficiency(revenue_series, growth_series)

    # Compute payback estimate (months to recover average monthly revenue)
    payback_estimate: float | None = None
    if revenue is not None and len(revenue_series) >= 2:
        avg_revenue = sum(revenue_series) / len(revenue_series)
        if avg_revenue > _ZERO_GUARD:
            payback_estimate = round(revenue / avg_revenue, 2)

    # Build estimated metrics — merge with any existing explicit metrics
    estimated_metrics = dict(metrics)
    estimation_sources: dict[str, str] = {}

    if estimated_metrics.get("churn_rate") is None and implied_churn is not None:
        estimated_metrics["churn_rate"] = round(implied_churn, 6)
        estimation_sources["churn_rate"] = "revenue_implied"
    if estimated_metrics.get("ltv") is None and estimated_ltv is not None:
        estimated_metrics["ltv"] = estimated_ltv
        estimation_sources["ltv"] = "revenue_derived"
    if payback_estimate is not None:
        estimated_metrics["payback_months_estimate"] = payback_estimate
        estimation_sources["payback_months_estimate"] = "revenue_derived"

    # Generate signals from estimated metrics
    signals = _evaluate_signals(
        estimated_metrics,
        trends,
        config=cfg,
        has_trend_data=has_trend_data,
    )

    # Add growth efficiency signals
    burn_risk = efficiency.get("burn_risk_indicator")
    if burn_risk == "high":
        signals.append(UnitEconomicsSignal(
            signal="burn_risk_elevated",
            severity="critical",
            confidence=0.7,
            description=(
                "Revenue trajectory signals elevated burn risk: "
                f"decline ratio {efficiency.get('decline_ratio', 0):.0%} "
                f"with growth CV {efficiency.get('growth_stability_cv', 0):.2f}."
            ),
            evidence=efficiency,
        ))
    elif burn_risk == "moderate":
        signals.append(UnitEconomicsSignal(
            signal="burn_risk_moderate",
            severity="warning",
            confidence=0.6,
            description=(
                "Revenue patterns indicate moderate burn risk. "
                "Growth is inconsistent or partially declining."
            ),
            evidence=efficiency,
        ))

    velocity = efficiency.get("revenue_velocity")
    if velocity is not None:
        if velocity < -0.05:
            signals.append(UnitEconomicsSignal(
                signal="revenue_contraction",
                severity="critical",
                confidence=0.8,
                description=f"Average period revenue change is {velocity:.1%}, indicating contraction.",
                evidence={"revenue_velocity": velocity},
            ))
        elif velocity > 0.05:
            signals.append(UnitEconomicsSignal(
                signal="revenue_expansion",
                severity="info",
                confidence=0.8,
                description=f"Average period revenue change is +{velocity:.1%}, indicating expansion.",
                evidence={"revenue_velocity": velocity},
            ))

    estimation_warnings = list(warnings)
    estimation_warnings.append(
        "Unit economics estimated from revenue trajectory (fallback mode). "
        "Confidence reduced."
    )
    if implied_churn is not None:
        estimation_warnings.append(
            f"Churn rate implied from revenue decline pattern: {implied_churn:.2%}."
        )
    if estimated_ltv is not None:
        estimation_warnings.append(
            f"LTV estimated from revenue / implied churn: {estimated_ltv:.2f}."
        )

    severity_rank = {"critical": 3, "warning": 2, "info": 1}
    signal_summary = (
        max(signals, key=lambda s: severity_rank.get(s.severity, 0)).signal
        if signals
        else "estimated_from_revenue"
    )

    # Confidence: lower than explicit metrics, penalized for estimation
    available_estimated = sum(1 for v in estimated_metrics.values() if v is not None)
    base_confidence = 0.5 if signals else 0.3
    completeness = min(1.0, available_estimated / 5.0)
    estimation_penalty = 0.7  # 30% penalty for estimated metrics
    confidence = round(base_confidence * completeness * estimation_penalty, 6)

    return {
        "metrics": estimated_metrics,
        "metric_series": metric_series,
        "trends": trends,
        "signals": [s.as_dict() for s in signals],
        "signal_summary": signal_summary,
        "confidence": confidence,
        "available_metrics": available_estimated,
        "estimation_method": "revenue_trajectory_fallback",
        "estimation_sources": estimation_sources,
        "growth_efficiency": efficiency,
        "status": "success",
        "warnings": estimation_warnings,
        "config": {
            "ltv_cac_healthy": cfg.ltv_cac_healthy,
            "ltv_cac_marginal": cfg.ltv_cac_marginal,
            "churn_healthy": cfg.churn_healthy,
            "churn_critical": cfg.churn_critical,
            "revenue_growth_healthy": cfg.revenue_growth_healthy,
            "cac_increase_warning": cfg.cac_increase_warning,
        },
    }


def _evaluate_signals(
    metrics: dict[str, float | None],
    trends: dict[str, float | None],
    *,
    config: UnitEconomicsConfig,
    has_trend_data: bool,
) -> list[UnitEconomicsSignal]:
    signals: list[UnitEconomicsSignal] = []

    ltv = metrics.get("ltv")
    cac = metrics.get("cac")
    ltv_cac = metrics.get("ltv_cac_ratio")
    churn = metrics.get("churn_rate")
    growth = metrics.get("growth_rate")

    churn_trend = trends.get("churn_rate")
    cac_trend = trends.get("cac")
    ltv_trend = trends.get("ltv")

    if ltv_cac is not None:
        if ltv_cac < config.ltv_cac_critical:
            signals.append(
                UnitEconomicsSignal(
                    signal="acquisition_inefficiency",
                    severity="critical",
                    confidence=0.95,
                    description=(
                        f"LTV/CAC ratio is {ltv_cac:.2f}x (< {config.ltv_cac_critical:.2f}x). "
                        "Customer acquisition is value-destructive."
                    ),
                    evidence={"ltv_cac_ratio": round(ltv_cac, 6), "ltv": ltv, "cac": cac},
                )
            )
        elif ltv_cac < config.ltv_cac_healthy:
            signals.append(
                UnitEconomicsSignal(
                    signal="payback_risk",
                    severity="warning",
                    confidence=0.8,
                    description=(
                        f"LTV/CAC ratio is {ltv_cac:.2f}x (between "
                        f"{config.ltv_cac_marginal:.2f}x and {config.ltv_cac_healthy:.2f}x)."
                    ),
                    evidence={"ltv_cac_ratio": round(ltv_cac, 6)},
                )
            )
        else:
            signals.append(
                UnitEconomicsSignal(
                    signal="unit_economics_healthy",
                    severity="info",
                    confidence=0.9,
                    description=(
                        f"LTV/CAC ratio is {ltv_cac:.2f}x (>= {config.ltv_cac_healthy:.2f}x). "
                        "Acquisition economics are healthy."
                    ),
                    evidence={"ltv_cac_ratio": round(ltv_cac, 6)},
                )
            )

    if churn is not None and churn >= config.churn_critical:
        signals.append(
            UnitEconomicsSignal(
                signal="churn_crisis",
                severity="critical",
                confidence=0.9,
                description=(
                    f"Churn rate is {churn:.1%} (>= {config.churn_critical:.0%}), indicating severe attrition."
                ),
                evidence={"churn_rate": round(churn, 6)},
            )
        )

    if has_trend_data and growth is not None:
        rising_attrition = churn_trend is not None and churn_trend > 0.0
        rising_cac_vs_ltv = (
            cac_trend is not None
            and cac_trend > config.cac_increase_warning
            and (ltv_trend is None or ltv_trend <= cac_trend)
        )
        if growth >= config.revenue_growth_healthy and (rising_attrition or rising_cac_vs_ltv):
            evidence: dict[str, Any] = {"growth_rate": round(growth, 6)}
            if churn_trend is not None:
                evidence["churn_trend"] = round(churn_trend, 6)
            if cac_trend is not None:
                evidence["cac_trend"] = round(cac_trend, 6)
            if ltv_trend is not None:
                evidence["ltv_trend"] = round(ltv_trend, 6)
            signals.append(
                UnitEconomicsSignal(
                    signal="unsustainable_growth",
                    severity="warning",
                    confidence=0.8,
                    description=(
                        "Revenue is growing, but acquisition or retention trends are deteriorating."
                    ),
                    evidence=evidence,
                )
            )

    if growth is not None and churn is not None and ltv_cac is not None:
        if (
            growth >= config.revenue_growth_healthy
            and churn <= config.churn_healthy
            and ltv_cac >= config.ltv_cac_healthy
        ):
            signals.append(
                UnitEconomicsSignal(
                    signal="healthy_growth",
                    severity="info",
                    confidence=0.9,
                    description=(
                        f"Growth ({growth:.1%}), churn ({churn:.1%}), and LTV/CAC ({ltv_cac:.2f}x) are healthy."
                    ),
                    evidence={
                        "growth_rate": round(growth, 6),
                        "churn_rate": round(churn, 6),
                        "ltv_cac_ratio": round(ltv_cac, 6),
                    },
                )
            )

    return signals


def analyze_unit_economics(
    records: Sequence[Mapping[str, Any]],
    *,
    business_type: str = "",
    config: UnitEconomicsConfig | None = None,
) -> dict[str, Any]:
    """
    Analyze unit economics from KPI records and generate deterministic signals.

    The returned payload is JSON-compatible and designed to be wrapped in node
    result envelopes.
    """
    cfg = config or UnitEconomicsConfig()
    if not records:
        return _insufficient_result("No KPI records provided.")

    ordered_records = _sort_records(records)
    aliases = _aliases_for_business_type(business_type)

    latest = ordered_records[-1]
    latest_kpis = latest.get("computed_kpis")
    if not isinstance(latest_kpis, Mapping):
        return _insufficient_result("Latest record has no computed_kpis.")

    ltv = _extract_metric(latest_kpis, aliases.get("ltv", ()))
    cac = _extract_metric(latest_kpis, aliases.get("cac", ()))
    churn = _extract_metric(latest_kpis, aliases.get("churn_rate", ()))
    revenue = _extract_metric(latest_kpis, aliases.get("revenue", ()))
    revenue_per_customer = _extract_metric(latest_kpis, aliases.get("revenue_per_customer", ()))
    growth = _extract_metric(latest_kpis, aliases.get("growth_rate", ()))

    if revenue_per_customer is None and revenue is not None:
        active_customers = _extract_metric(
            latest_kpis,
            ("active_customer_count", "active_customers", "unique_customers"),
        )
        revenue_per_customer = compute_revenue_per_customer(revenue, active_customers)

    ltv_cac = compute_ltv_cac_ratio(ltv, cac)

    metrics: dict[str, float | None] = {
        "ltv": _round_or_none(ltv),
        "cac": _round_or_none(cac),
        "ltv_cac_ratio": _round_or_none(ltv_cac),
        "churn_rate": _round_or_none(churn),
        "revenue_per_customer": _round_or_none(revenue_per_customer),
        "revenue": _round_or_none(revenue),
        "growth_rate": _round_or_none(growth),
    }

    metric_series: dict[str, list[float]] = {}
    for canonical_metric, metric_aliases in aliases.items():
        series = _extract_series(ordered_records, metric_aliases)
        if series:
            metric_series[canonical_metric] = [round(v, 6) for v in series]

    trends: dict[str, float | None] = {}
    has_trend_data = False
    for metric_name, series in metric_series.items():
        if len(series) >= 2:
            trend = _pct_change(series[-1], series[-2])
            trends[metric_name] = _round_or_none(trend)
            if trend is not None:
                has_trend_data = True
        else:
            trends[metric_name] = None

    warnings: list[str] = []
    if ltv is None:
        warnings.append("LTV unavailable.")
    if cac is None:
        warnings.append("CAC unavailable; LTV/CAC ratio cannot be computed.")
    if churn is None:
        warnings.append("Churn rate unavailable.")
    if revenue_per_customer is None:
        warnings.append("Revenue per customer unavailable.")
    if not has_trend_data:
        warnings.append("Insufficient history for trend-based signals.")

    available_core_metrics = sum(
        1
        for value in (
            metrics["ltv"],
            metrics["cac"],
            metrics["ltv_cac_ratio"],
            metrics["churn_rate"],
            metrics["revenue_per_customer"],
        )
        if value is not None
    )
    if available_core_metrics < 2:
        # Fallback: estimate from revenue trajectory
        estimated = _estimate_unit_economics_from_revenue(
            ordered_records=ordered_records,
            aliases=aliases,
            metrics=metrics,
            metric_series=metric_series,
            trends=trends,
            has_trend_data=has_trend_data,
            warnings=warnings,
            cfg=cfg,
        )
        if estimated is not None:
            return estimated

        result = _insufficient_result(
            f"Only {available_core_metrics} core metric(s) available; need at least 2.",
        )
        result["metrics"] = metrics
        result["warnings"] = warnings
        return result

    signals = _evaluate_signals(
        metrics,
        trends,
        config=cfg,
        has_trend_data=has_trend_data,
    )
    severity_rank = {"critical": 3, "warning": 2, "info": 1}
    signal_summary = (
        max(signals, key=lambda item: severity_rank.get(item.severity, 0)).signal
        if signals
        else "no_signals"
    )

    if signals:
        base_confidence = min(signal.confidence for signal in signals)
    else:
        base_confidence = 0.5
    completeness = available_core_metrics / 5.0
    trend_factor = 1.0 if has_trend_data else 0.9
    confidence = round(base_confidence * completeness * trend_factor, 6)

    return {
        "metrics": metrics,
        "metric_series": metric_series,
        "trends": trends,
        "signals": [signal.as_dict() for signal in signals],
        "signal_summary": signal_summary,
        "confidence": confidence,
        "available_metrics": available_core_metrics,
        "status": "success",
        "warnings": warnings,
        "config": {
            "ltv_cac_healthy": cfg.ltv_cac_healthy,
            "ltv_cac_marginal": cfg.ltv_cac_marginal,
            "churn_healthy": cfg.churn_healthy,
            "churn_critical": cfg.churn_critical,
            "revenue_growth_healthy": cfg.revenue_growth_healthy,
            "cac_increase_warning": cfg.cac_increase_warning,
        },
    }
