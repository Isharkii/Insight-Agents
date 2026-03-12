"""
app/services/statistics/survival_analysis.py

Kaplan–Meier survival estimator for cohort retention analysis.

Correctly handles right-censored observations (customers still active)
to produce unbiased survival curves and median lifetime estimates.

All math uses only the Python standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

_ZERO_GUARD = 1e-9


@dataclass(frozen=True)
class SurvivalConfig:
    """Configuration for survival analysis."""

    min_subjects: int = 5
    confidence_level: float = 0.95  # for Greenwood CI


def kaplan_meier(
    durations: Sequence[float | int],
    observed: Sequence[bool],
    *,
    config: SurvivalConfig | None = None,
) -> dict[str, Any]:
    """
    Compute the Kaplan–Meier survival function.

    Parameters
    ----------
    durations:
        Time-to-event for each subject (e.g., months active).
    observed:
        True if the event (churn) was observed, False if right-censored
        (still active at end of study).
    config:
        Optional thresholds.

    Returns
    -------
    dict
        survival_curve, median_survival, mean_survival, confidence_bands,
        at_risk_table, summary, warnings
    """
    cfg = config or SurvivalConfig()
    warnings: list[str] = []

    n = min(len(durations), len(observed))
    if n < cfg.min_subjects:
        return {
            "status": "insufficient_data",
            "survival_curve": [],
            "median_survival": None,
            "mean_survival": None,
            "confidence_bands": [],
            "at_risk_table": [],
            "summary": {
                "total_subjects": n,
                "total_events": 0,
                "total_censored": 0,
                "censoring_rate": 0.0,
            },
            "warnings": [
                f"Need at least {cfg.min_subjects} subjects; got {n}."
            ],
        }

    # Build event table sorted by time
    subjects = sorted(
        zip(
            (float(d) for d in durations[:n]),
            (bool(o) for o in observed[:n]),
        ),
        key=lambda pair: pair[0],
    )

    total_events = sum(1 for _, o in subjects if o)
    total_censored = n - total_events

    # Group by distinct times
    event_times: list[float] = []
    events_at: dict[float, int] = {}
    censored_at: dict[float, int] = {}

    for dur, obs in subjects:
        t = round(dur, 6)
        if t not in events_at:
            event_times.append(t)
            events_at[t] = 0
            censored_at[t] = 0
        if obs:
            events_at[t] += 1
        else:
            censored_at[t] += 1

    event_times.sort()

    # Kaplan–Meier estimator
    survival_curve: list[dict[str, Any]] = []
    confidence_bands: list[dict[str, Any]] = []
    at_risk_table: list[dict[str, Any]] = []

    n_at_risk = n
    survival_prob = 1.0
    greenwood_sum = 0.0  # Σ d_i / (n_i · (n_i - d_i))

    # Add time 0
    survival_curve.append({"time": 0.0, "survival": 1.0})

    z = _z_for_confidence(cfg.confidence_level)

    for t in event_times:
        d = events_at[t]     # events at this time
        c = censored_at[t]   # censored at this time

        at_risk_table.append({
            "time": t,
            "at_risk": n_at_risk,
            "events": d,
            "censored": c,
        })

        if d > 0 and n_at_risk > 0:
            # S(t) = S(t-1) × (1 - d/n)
            hazard = d / n_at_risk
            survival_prob *= (1.0 - hazard)

            # Greenwood variance: Var(S) = S² × Σ d_i / (n_i(n_i - d_i))
            denom = n_at_risk * (n_at_risk - d)
            if denom > 0:
                greenwood_sum += d / denom

        survival_curve.append({
            "time": t,
            "survival": round(max(0.0, survival_prob), 6),
        })

        # Greenwood confidence interval
        if survival_prob > _ZERO_GUARD and greenwood_sum >= 0:
            se = survival_prob * math.sqrt(greenwood_sum)
            lower = max(0.0, survival_prob - z * se)
            upper = min(1.0, survival_prob + z * se)
        else:
            lower = 0.0
            upper = min(1.0, survival_prob)
            se = 0.0

        confidence_bands.append({
            "time": t,
            "lower": round(lower, 6),
            "upper": round(upper, 6),
            "standard_error": round(se, 6),
        })

        # Remove events and censored from risk set
        n_at_risk -= (d + c)

    # Median survival: smallest t where S(t) ≤ 0.5
    median_survival = _compute_median(survival_curve)

    # Restricted mean survival time (area under curve)
    mean_survival = _restricted_mean(survival_curve)

    if total_censored > 0:
        censoring_rate = total_censored / n
        if censoring_rate > 0.5:
            warnings.append(
                f"High censoring rate ({censoring_rate:.1%}); "
                f"survival estimates may be imprecise."
            )

    return {
        "status": "success",
        "survival_curve": survival_curve,
        "median_survival": round(median_survival, 6) if median_survival is not None else None,
        "mean_survival": round(mean_survival, 6) if mean_survival is not None else None,
        "confidence_bands": confidence_bands,
        "at_risk_table": at_risk_table,
        "summary": {
            "total_subjects": n,
            "total_events": total_events,
            "total_censored": total_censored,
            "censoring_rate": round(total_censored / max(n, 1), 6),
        },
        "warnings": warnings,
    }


def survival_from_retention_curve(
    retention_rates: Sequence[float],
    *,
    total_customers: int | None = None,
    config: SurvivalConfig | None = None,
) -> dict[str, Any]:
    """
    Convert aggregate retention rates into a Kaplan–Meier survival estimate.

    This bridges the existing cohort analytics (which produces retention
    curves) with proper survival analysis.  Retention rates are treated
    as the fraction surviving at each period.

    Parameters
    ----------
    retention_rates:
        Ordered retention rates (period 0 = 1.0 or initial, period 1, ...).
    total_customers:
        Optional total cohort size for at-risk computation.

    Returns
    -------
    dict with survival_curve, median_survival, mean_survival, hazard_rates
    """
    cfg = config or SurvivalConfig()
    rates = [max(0.0, min(1.0, float(r))) for r in retention_rates if _is_finite(r)]

    if len(rates) < 2:
        return {
            "status": "insufficient_data",
            "survival_curve": [],
            "median_survival": None,
            "mean_survival": None,
            "hazard_rates": [],
            "warnings": ["Need at least 2 retention periods."],
        }

    # Normalize: if first value is not ~1.0, scale all rates
    if rates[0] > _ZERO_GUARD:
        scale = 1.0 / rates[0]
        survival_values = [min(1.0, r * scale) for r in rates]
    else:
        survival_values = list(rates)

    # Build survival curve
    survival_curve = [
        {"time": float(i), "survival": round(s, 6)}
        for i, s in enumerate(survival_values)
    ]

    # Compute period hazard rates: h(t) = 1 - S(t)/S(t-1)
    hazard_rates: list[dict[str, Any]] = []
    for i in range(1, len(survival_values)):
        prev = survival_values[i - 1]
        curr = survival_values[i]
        if prev > _ZERO_GUARD:
            h = 1.0 - (curr / prev)
        else:
            h = 0.0
        hazard_rates.append({
            "period": i,
            "hazard_rate": round(max(0.0, h), 6),
            "cumulative_hazard": round(-math.log(max(curr, _ZERO_GUARD)), 6),
        })

    median_survival = _compute_median(survival_curve)
    mean_survival = _restricted_mean(survival_curve)

    # Hazard trend: is churn accelerating?
    hazard_values = [h["hazard_rate"] for h in hazard_rates]
    hazard_trend = _hazard_trend(hazard_values)

    warnings: list[str] = []
    if hazard_trend == "increasing":
        warnings.append("Hazard rate increasing over time — churn is accelerating.")

    return {
        "status": "success",
        "survival_curve": survival_curve,
        "median_survival": round(median_survival, 6) if median_survival is not None else None,
        "mean_survival": round(mean_survival, 6) if mean_survival is not None else None,
        "hazard_rates": hazard_rates,
        "hazard_trend": hazard_trend,
        "warnings": warnings,
    }


def _compute_median(survival_curve: list[dict[str, Any]]) -> float | None:
    """Smallest time where survival ≤ 0.5."""
    for point in survival_curve:
        if point["survival"] <= 0.5:
            return float(point["time"])
    return None


def _restricted_mean(survival_curve: list[dict[str, Any]]) -> float | None:
    """Area under the survival curve (trapezoidal rule)."""
    if len(survival_curve) < 2:
        return None

    area = 0.0
    for i in range(1, len(survival_curve)):
        dt = survival_curve[i]["time"] - survival_curve[i - 1]["time"]
        avg_s = (survival_curve[i]["survival"] + survival_curve[i - 1]["survival"]) / 2.0
        area += dt * avg_s

    return area


def _hazard_trend(hazard_values: list[float]) -> str:
    """Classify hazard rate trend."""
    if len(hazard_values) < 3:
        return "insufficient_data"

    # Simple slope via first vs last third
    third = max(1, len(hazard_values) // 3)
    early = sum(hazard_values[:third]) / third
    late = sum(hazard_values[-third:]) / third

    diff = late - early
    if diff > 0.02:
        return "increasing"
    elif diff < -0.02:
        return "decreasing"
    return "stable"


def _z_for_confidence(level: float) -> float:
    """Approximate z-score for given confidence level."""
    # Common values
    if level >= 0.99:
        return 2.576
    if level >= 0.95:
        return 1.96
    if level >= 0.90:
        return 1.645
    return 1.28


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
