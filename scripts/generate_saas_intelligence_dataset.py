"""Generate a synthetic SaaS intelligence dataset for stress testing analytics.

Output:
    data/saas_intelligence_dataset.csv
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


SEED = 20260302
MONTHS = 36
START_MONTH = date(2023, 1, 1)
OUTPUT_PATH = Path("data/saas_intelligence_dataset.csv")


@dataclass
class EntityConfig:
    name: str
    pattern: str
    initial_active: int
    seasonal_amplitude: float
    base_churn: float
    churn_noise: float


@dataclass
class Customer:
    customer_id: str
    signup_date: date
    region: str
    role: str
    team: str
    plan_tier: str
    seats: int
    last_mrr: float
    active: bool = True


ENTITY_CONFIGS: tuple[EntityConfig, ...] = (
    EntityConfig(
        name="STEADYFLOW_CLOUD",
        pattern="steady",
        initial_active=170,
        seasonal_amplitude=0.03,
        base_churn=0.018,
        churn_noise=0.004,
    ),
    EntityConfig(
        name="VOLTSTACK_ANALYTICS",
        pattern="volatile",
        initial_active=150,
        seasonal_amplitude=0.07,
        base_churn=0.040,
        churn_noise=0.010,
    ),
    EntityConfig(
        name="PLATEAUWORKS_SAAS",
        pattern="plateau_decline",
        initial_active=180,
        seasonal_amplitude=0.04,
        base_churn=0.024,
        churn_noise=0.006,
    ),
)

REGIONS: tuple[tuple[str, float], ...] = (
    ("NA", 0.48),
    ("EU", 0.27),
    ("APAC", 0.18),
    ("LATAM", 0.07),
)

ROLES: tuple[tuple[str, float], ...] = (
    ("individual_contributor", 0.52),
    ("team_lead", 0.25),
    ("operations_manager", 0.15),
    ("executive_sponsor", 0.08),
)

TEAMS: tuple[tuple[str, float], ...] = (
    ("sales", 0.34),
    ("marketing", 0.22),
    ("customer_success", 0.24),
    ("product", 0.20),
)

PLAN_TIERS: tuple[tuple[str, tuple[int, int], tuple[int, int], float], ...] = (
    ("starter", (70, 190), (1, 5), 0.44),
    ("growth", (190, 620), (6, 20), 0.40),
    ("enterprise", (620, 2400), (20, 110), 0.16),
)

# Occasional anomalies per entity by month index.
ANOMALY_FACTORS: dict[str, dict[int, float]] = {
    "STEADYFLOW_CLOUD": {14: 1.12, 28: 0.90},
    "VOLTSTACK_ANALYTICS": {9: 1.34, 20: 0.72, 31: 1.28},
    "PLATEAUWORKS_SAAS": {11: 1.15, 27: 0.80},
}

CHURN_SHOCKS: dict[str, dict[int, float]] = {
    "STEADYFLOW_CLOUD": {28: 0.010},
    "VOLTSTACK_ANALYTICS": {20: 0.035, 31: -0.010},
    "PLATEAUWORKS_SAAS": {27: 0.020, 30: 0.015},
}


def add_months(d: date, months: int) -> date:
    month_index = (d.year * 12 + (d.month - 1)) + months
    year = month_index // 12
    month = month_index % 12 + 1
    return date(year, month, 1)


def iso_months(start: date, count: int) -> list[date]:
    return [add_months(start, idx) for idx in range(count)]


def weighted_choice(items: Iterable[tuple[str, float]]) -> str:
    choices = list(items)
    labels = [label for label, _ in choices]
    weights = [weight for _, weight in choices]
    return random.choices(labels, weights=weights, k=1)[0]


def choose_plan() -> tuple[str, float, int]:
    tiers = [(name, probability) for name, _, _, probability in PLAN_TIERS]
    plan = weighted_choice(tiers)
    tier_cfg = next(cfg for cfg in PLAN_TIERS if cfg[0] == plan)
    mrr_low, mrr_high = tier_cfg[1]
    seats_low, seats_high = tier_cfg[2]
    base_mrr = round(random.uniform(mrr_low, mrr_high), 2)
    seats = random.randint(seats_low, seats_high)
    return plan, base_mrr, seats


def growth_rate(pattern: str, month_idx: int) -> float:
    if pattern == "steady":
        base = 0.016 + 0.005 * math.sin((2 * math.pi * month_idx) / 12.0)
        return base + random.gauss(0.0, 0.004)
    if pattern == "volatile":
        base = 0.010 + 0.045 * math.sin((2 * math.pi * month_idx) / 6.0)
        return base + random.gauss(0.0, 0.018)
    if month_idx < 12:
        return 0.022 + random.gauss(0.0, 0.006)
    if month_idx < 24:
        return 0.004 + random.gauss(0.0, 0.005)
    return -0.018 + random.gauss(0.0, 0.007)


def make_customer(entity_name: str, signup_date: date, serial: int) -> Customer:
    region = weighted_choice(REGIONS)
    role = weighted_choice(ROLES)
    team = weighted_choice(TEAMS)
    plan_tier, base_mrr, seats = choose_plan()
    customer_id = f"{entity_name[:6]}_{serial:05d}"
    return Customer(
        customer_id=customer_id,
        signup_date=signup_date,
        region=region,
        role=role,
        team=team,
        plan_tier=plan_tier,
        seats=seats,
        last_mrr=base_mrr,
    )


def generate() -> list[dict[str, str | int | float]]:
    random.seed(SEED)
    months = iso_months(START_MONTH, MONTHS)
    rows: list[dict[str, str | int | float]] = []

    for cfg in ENTITY_CONFIGS:
        customers: dict[str, Customer] = {}
        active_ids: set[str] = set()
        serial = 0

        # Seed initial active base with signup dates across the prior year.
        for _ in range(cfg.initial_active):
            serial += 1
            historical_signup = add_months(START_MONTH, -random.randint(0, 11))
            customer = make_customer(cfg.name, historical_signup, serial)
            customers[customer.customer_id] = customer
            active_ids.add(customer.customer_id)

        for month_idx, ts in enumerate(months):
            anomaly_factor = ANOMALY_FACTORS.get(cfg.name, {}).get(month_idx, 1.0)
            churn_shock = CHURN_SHOCKS.get(cfg.name, {}).get(month_idx, 0.0)
            seasonal_index = 1.0 + cfg.seasonal_amplitude * math.sin(
                (2.0 * math.pi * (month_idx % 12)) / 12.0
            )

            active_before = [customers[cid] for cid in sorted(active_ids)]
            churn_rate = max(
                0.005,
                min(
                    0.20,
                    cfg.base_churn
                    + 0.01 * (1.0 - seasonal_index)
                    + churn_shock
                    + random.gauss(0.0, cfg.churn_noise),
                ),
            )
            churn_count = min(len(active_before), int(round(len(active_before) * churn_rate)))

            churned_customers: list[Customer] = []
            if churn_count > 0:
                churned_customers = random.sample(active_before, churn_count)
                for customer in churned_customers:
                    active_ids.discard(customer.customer_id)
                    customer.active = False
                    rows.append(
                        {
                            "entity_name": cfg.name,
                            "timestamp": ts.isoformat(),
                            "customer_id": customer.customer_id,
                            "signup_date": customer.signup_date.isoformat(),
                            "cohort_month": customer.signup_date.replace(day=1).isoformat(),
                            "region": customer.region,
                            "role": customer.role,
                            "team": customer.team,
                            "plan_tier": customer.plan_tier,
                            "lifecycle_status": "churned",
                            "mrr_usd": 0.0,
                            "seats": customer.seats,
                            "expansion_mrr_usd": 0.0,
                            "contraction_mrr_usd": round(customer.last_mrr, 2),
                            "churned_this_month": 1,
                            "new_customer_this_month": 0,
                            "seasonality_index": round(seasonal_index, 4),
                            "noise_component": 0.0,
                            "anomaly_flag": int(anomaly_factor != 1.0),
                            "category": "saas",
                            "source_type": "csv",
                        }
                    )

            # Fill/adjust active base according to target growth path.
            g_rate = growth_rate(cfg.pattern, month_idx)
            desired_active = max(60, int(round(len(active_ids) * (1.0 + g_rate))))
            new_needed = max(0, desired_active - len(active_ids))
            if month_idx % 6 == 0:
                new_needed = max(new_needed, 3)

            for _ in range(new_needed):
                serial += 1
                customer = make_customer(cfg.name, ts, serial)
                customers[customer.customer_id] = customer
                active_ids.add(customer.customer_id)

            # Emit active monthly snapshots.
            for cid in sorted(active_ids):
                customer = customers[cid]
                tenure_months = max(
                    0,
                    (ts.year - customer.signup_date.year) * 12
                    + (ts.month - customer.signup_date.month),
                )

                if cfg.pattern == "steady":
                    drift = 0.004 + random.gauss(0.0, 0.010)
                elif cfg.pattern == "volatile":
                    drift = 0.002 + random.gauss(0.0, 0.040)
                elif month_idx < 12:
                    drift = 0.006 + random.gauss(0.0, 0.014)
                elif month_idx < 24:
                    drift = 0.001 + random.gauss(0.0, 0.012)
                else:
                    drift = -0.008 + random.gauss(0.0, 0.015)

                tenure_uplift = min(0.10, tenure_months * 0.002)
                noise = random.gauss(0.0, 0.020)
                next_mrr = customer.last_mrr * (1.0 + drift + tenure_uplift * 0.15 + noise)
                next_mrr = next_mrr * seasonal_index * anomaly_factor
                next_mrr = max(45.0, next_mrr)
                next_mrr = round(next_mrr, 2)

                expansion = max(0.0, round(next_mrr - customer.last_mrr, 2))
                contraction = max(0.0, round(customer.last_mrr - next_mrr, 2))
                is_new = int(customer.signup_date == ts)

                rows.append(
                    {
                        "entity_name": cfg.name,
                        "timestamp": ts.isoformat(),
                        "customer_id": customer.customer_id,
                        "signup_date": customer.signup_date.isoformat(),
                        "cohort_month": customer.signup_date.replace(day=1).isoformat(),
                        "region": customer.region,
                        "role": customer.role,
                        "team": customer.team,
                        "plan_tier": customer.plan_tier,
                        "lifecycle_status": "active",
                        "mrr_usd": next_mrr,
                        "seats": customer.seats,
                        "expansion_mrr_usd": expansion,
                        "contraction_mrr_usd": contraction,
                        "churned_this_month": 0,
                        "new_customer_this_month": is_new,
                        "seasonality_index": round(seasonal_index, 4),
                        "noise_component": round(noise, 4),
                        "anomaly_flag": int(anomaly_factor != 1.0),
                        "category": "saas",
                        "source_type": "csv",
                    }
                )
                customer.last_mrr = next_mrr
                customer.active = True

    return rows


def validate_no_duplicate_monthly_keys(rows: list[dict[str, str | int | float]]) -> None:
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (
            str(row["entity_name"]),
            str(row["customer_id"]),
            str(row["timestamp"]),
        )
        if key in seen:
            raise ValueError(f"Duplicate entity/customer/timestamp detected: {key}")
        seen.add(key)


def save_csv(rows: list[dict[str, str | int | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "entity_name",
        "timestamp",
        "customer_id",
        "signup_date",
        "cohort_month",
        "region",
        "role",
        "team",
        "plan_tier",
        "lifecycle_status",
        "mrr_usd",
        "seats",
        "expansion_mrr_usd",
        "contraction_mrr_usd",
        "churned_this_month",
        "new_customer_this_month",
        "seasonality_index",
        "noise_component",
        "anomaly_flag",
        "category",
        "source_type",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str | int | float]]) -> None:
    timestamps = sorted({str(row["timestamp"]) for row in rows})
    rows_per_entity: dict[str, int] = {}
    customers_per_entity: dict[str, set[str]] = {}

    for row in rows:
        entity = str(row["entity_name"])
        rows_per_entity[entity] = rows_per_entity.get(entity, 0) + 1
        customers_per_entity.setdefault(entity, set()).add(str(row["customer_id"]))

    print("Distinct timestamps:", len(timestamps), f"({timestamps[0]} -> {timestamps[-1]})")
    print("Rows per entity:")
    for entity in sorted(rows_per_entity):
        print(f"  - {entity}: {rows_per_entity[entity]}")
    print("Sample of 5 rows:")
    for row in rows[:5]:
        print(f"  - {row}")
    print("Total customers per entity:")
    for entity in sorted(customers_per_entity):
        print(f"  - {entity}: {len(customers_per_entity[entity])}")


def main() -> None:
    rows = generate()
    validate_no_duplicate_monthly_keys(rows)
    save_csv(rows, OUTPUT_PATH)
    print(f"Saved dataset: {OUTPUT_PATH}")
    print_summary(rows)


if __name__ == "__main__":
    main()
