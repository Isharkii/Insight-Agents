"""
Generate ingestion-ready CSV data that satisfies risk-signal sufficiency.

Output schema:
    category,entity_name,metric_name,metric_value,timestamp,source_type

This script encodes two layers of sufficiency:
1) KPI gate minimums (from agent/signal_envelope.py)
2) Risk signal readiness (from agent/signal_normalizer.py families + depth)
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path
import random


MIN_SERIES_DEPTH_FOR_RISK = 2

# From agent/signal_envelope.py
KPI_MINIMUM_BY_BUSINESS_TYPE: dict[str, tuple[str, ...]] = {
    "saas": ("mrr", "churn_rate"),
    "ecommerce": ("revenue", "conversion_rate"),
    "agency": ("total_revenue", "client_churn"),
    "general_timeseries": ("revenue", "churn_rate"),
    "generic_timeseries": ("revenue", "churn_rate"),
}

# From agent/signal_normalizer.py families
RISK_SIGNAL_FAMILIES: dict[str, tuple[str, ...]] = {
    "revenue_family": (
        "revenue_growth_delta",
        "growth_rate",
        "revenue",
        "mrr",
        "total_revenue",
        "retainer_revenue",
    ),
    "churn_family": (
        "churn_delta",
        "churn_rate",
        "client_churn",
    ),
    "conversion_family": (
        "conversion_delta",
        "conversion_rate",
    ),
}

# Metrics to emit per business type (must include minimums + one from each risk family).
EMITTED_METRICS_BY_BUSINESS_TYPE: dict[str, tuple[str, ...]] = {
    "saas": ("mrr", "churn_rate", "conversion_rate", "growth_rate", "arpu", "ltv"),
    "ecommerce": ("revenue", "conversion_rate", "churn_rate", "aov", "cac", "purchase_frequency"),
    "agency": ("total_revenue", "client_churn", "conversion_rate", "utilization_rate", "revenue_per_employee"),
    "general_timeseries": ("revenue", "churn_rate", "conversion_rate"),
    "generic_timeseries": ("revenue", "churn_rate", "conversion_rate"),
}


def _month_starts(start_year: int, start_month: int, periods: int) -> list[date]:
    points: list[date] = []
    y = start_year
    m = start_month
    for _ in range(periods):
        points.append(date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return points


def _bounded(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _metric_value(metric: str, step: int, rng: random.Random) -> float:
    # Deterministic but non-flat series to avoid synthetic edge cases.
    if metric in {"mrr", "revenue", "total_revenue", "retainer_revenue"}:
        base = 100000.0 + (step * 1750.0)
        return round(base + rng.uniform(-600.0, 600.0), 2)
    if metric in {"churn_rate", "client_churn"}:
        base = 0.03 + (step * 0.0007)
        return round(_bounded(base + rng.uniform(-0.002, 0.002), 0.005, 0.2), 5)
    if metric == "conversion_rate":
        base = 0.045 + (step * 0.0005)
        return round(_bounded(base + rng.uniform(-0.003, 0.003), 0.005, 0.25), 5)
    if metric == "growth_rate":
        base = 0.015 + (step * 0.0003)
        return round(_bounded(base + rng.uniform(-0.01, 0.01), -0.2, 0.3), 5)
    if metric == "aov":
        return round(120.0 + (step * 1.3) + rng.uniform(-6.0, 6.0), 2)
    if metric == "cac":
        return round(340.0 + (step * 0.8) + rng.uniform(-15.0, 15.0), 2)
    if metric == "purchase_frequency":
        return round(_bounded(2.1 + (step * 0.02) + rng.uniform(-0.1, 0.1), 0.5, 8.0), 3)
    if metric == "utilization_rate":
        return round(_bounded(0.71 + (step * 0.001) + rng.uniform(-0.02, 0.02), 0.3, 0.98), 5)
    if metric == "revenue_per_employee":
        return round(8900.0 + (step * 35.0) + rng.uniform(-180.0, 180.0), 2)
    if metric == "arpu":
        return round(102.0 + (step * 0.25) + rng.uniform(-2.0, 2.0), 2)
    if metric == "ltv":
        return round(2300.0 + (step * 20.0) + rng.uniform(-90.0, 90.0), 2)
    return round(1.0 + step + rng.uniform(-0.2, 0.2), 5)


def _has_family(metrics: set[str], family: tuple[str, ...]) -> bool:
    return any(name in metrics for name in family)


def validate_sufficiency(business_type: str, periods: int, emitted_metrics: tuple[str, ...]) -> list[str]:
    problems: list[str] = []
    metric_set = set(emitted_metrics)

    if periods < MIN_SERIES_DEPTH_FOR_RISK:
        problems.append(
            f"periods={periods} is below minimum {MIN_SERIES_DEPTH_FOR_RISK} required for risk time-series depth"
        )

    minimum = KPI_MINIMUM_BY_BUSINESS_TYPE.get(business_type, KPI_MINIMUM_BY_BUSINESS_TYPE["general_timeseries"])
    missing_min = [m for m in minimum if m not in metric_set]
    if missing_min:
        problems.append(f"missing KPI minimum metrics for {business_type}: {missing_min}")

    for family_name, family_metrics in RISK_SIGNAL_FAMILIES.items():
        if not _has_family(metric_set, family_metrics):
            problems.append(f"missing required risk {family_name}: need one of {list(family_metrics)}")

    return problems


def generate_rows(
    *,
    business_type: str,
    entity_name: str,
    periods: int,
    start_year: int,
    start_month: int,
    source_type: str,
    seed: int,
) -> list[dict[str, str]]:
    category = "saas" if business_type == "saas" else (
        "ecommerce" if business_type == "ecommerce" else (
            "agency" if business_type == "agency" else "general_timeseries"
        )
    )
    metrics = EMITTED_METRICS_BY_BUSINESS_TYPE[business_type]
    rng = random.Random(seed)
    stamps = _month_starts(start_year, start_month, periods)

    rows: list[dict[str, str]] = []
    for step, ts in enumerate(stamps):
        for metric_name in metrics:
            value = _metric_value(metric_name, step, rng)
            rows.append(
                {
                    "category": category,
                    "entity_name": entity_name,
                    "metric_name": metric_name,
                    "metric_value": str(value),
                    "timestamp": ts.isoformat(),
                    "source_type": source_type,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate risk-sufficient ingestion CSV.")
    parser.add_argument("--business-type", choices=sorted(EMITTED_METRICS_BY_BUSINESS_TYPE.keys()), default="saas")
    parser.add_argument("--entity-name", default="RISK_READY_INC")
    parser.add_argument("--periods", type=int, default=6, help="Monthly points to generate (>=2 recommended).")
    parser.add_argument("--start-year", type=int, default=2025)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--source-type", default="internal_system")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="risk_sufficient_data.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    metrics = EMITTED_METRICS_BY_BUSINESS_TYPE[args.business_type]
    issues = validate_sufficiency(args.business_type, args.periods, metrics)
    if issues:
        raise SystemExit("Invalid generation config:\n- " + "\n- ".join(issues))

    rows = generate_rows(
        business_type=args.business_type,
        entity_name=args.entity_name,
        periods=args.periods,
        start_year=args.start_year,
        start_month=args.start_month,
        source_type=args.source_type,
        seed=args.seed,
    )

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "entity_name", "metric_name", "metric_value", "timestamp", "source_type"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Generated:", out_path)
    print("Rows:", len(rows))
    print("Business type:", args.business_type)
    print("Entity:", args.entity_name)
    print("Periods:", args.periods)
    print("Metrics per period:", len(metrics))
    print("KPI minimum required:", list(KPI_MINIMUM_BY_BUSINESS_TYPE[args.business_type]))
    print("Risk depth minimum:", MIN_SERIES_DEPTH_FOR_RISK)
    print("Risk family coverage metrics:", list(metrics))


if __name__ == "__main__":
    main()

