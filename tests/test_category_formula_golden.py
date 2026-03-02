from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pytest

from app.services.category_registry import require_category_pack, supported_categories
from app.services.kpi_orchestrator import KPIOrchestrator, _AggregatedInputs, _build_payload
from llm_synthesis.schema import InsightOutput


_BASE_AGG_INPUTS = _AggregatedInputs(
    subscription_revenues=[120.0, 180.0],
    active_customers=100,
    lost_customers=5,
    arpu=3.0,
    current_revenue=300.0,
    previous_revenue=240.0,
)


def _compute_payload(category: str, extra_inputs: Mapping[str, Any]) -> dict[str, Any]:
    pack = require_category_pack(category)
    metrics, validity = KPIOrchestrator()._compute(
        business_type=category,
        agg_inputs=_BASE_AGG_INPUTS,
        extra_inputs=dict(extra_inputs),
    )
    return _build_payload(metrics, validity, rate_metrics=pack.rate_metrics)


def _assert_metric(
    payload: Mapping[str, Any],
    metric_name: str,
    *,
    expected_value: float | None,
    expected_unit: str,
    expected_error: str | None,
) -> None:
    assert metric_name in payload
    entry = payload[metric_name]
    assert isinstance(entry, dict)
    actual_value = entry.get("value")
    if expected_value is None:
        assert actual_value is None
    else:
        assert actual_value == pytest.approx(expected_value)
    assert entry.get("unit") == expected_unit
    assert entry.get("error") == expected_error


@pytest.mark.parametrize(
    ("category", "extra_inputs", "expected"),
    [
        (
            "financial_markets",
            {"risk_free_rate": 0.05},
            {
                "market_revenue": (300.0, "currency", None),
                "account_churn": (0.05, "rate", None),
                "growth_rate": (0.25, "rate", None),
                "volatility": (30.0, "rate", None),
                "sharpe_like": ((0.25 - 0.05) / 30.0, "rate", None),
            },
        ),
        (
            "marketing_analytics",
            {"ad_spend": 150.0, "impressions": 10000, "clicks": 500},
            {
                "attributed_revenue": (300.0, "currency", None),
                "pipeline_churn": (0.05, "rate", None),
                "growth_rate": (0.25, "rate", None),
                "roas": (2.0, "currency", None),
                "ctr": (0.05, "rate", None),
                "cac": (1.5, "currency", None),
            },
        ),
        (
            "operations",
            {"labor_hours": 120.0, "downtime_hours": 10.0, "capacity_hours": 200.0},
            {
                "throughput": (300.0, "currency", None),
                "defect_rate": (0.05, "rate", None),
                "growth_rate": (0.25, "rate", None),
                "productivity": (2.5, "currency", None),
                "uptime_rate": (0.95, "rate", None),
            },
        ),
        (
            "retail",
            {"footfall": 1000.0, "orders": 200.0, "cogs": 180.0},
            {
                "net_sales": (300.0, "currency", None),
                "customer_churn": (0.05, "rate", None),
                "growth_rate": (0.25, "rate", None),
                "conversion_rate": (0.2, "rate", None),
                "gross_margin_rate": (0.4, "rate", None),
                "average_ticket": (1.5, "currency", None),
            },
        ),
        (
            "healthcare",
            {"occupied_beds": 80.0, "total_beds": 100.0, "staff_hours": 400.0},
            {
                "patient_revenue": (300.0, "currency", None),
                "readmission_rate": (0.05, "rate", None),
                "growth_rate": (0.25, "rate", None),
                "bed_occupancy_rate": (0.8, "rate", None),
                "revenue_per_staff_hour": (0.75, "currency", None),
            },
        ),
    ],
)
def test_golden_outputs_for_new_categories(
    category: str,
    extra_inputs: Mapping[str, Any],
    expected: Mapping[str, tuple[float | None, str, str | None]],
) -> None:
    payload = _compute_payload(category, extra_inputs)
    for metric_name, (value, unit, error) in expected.items():
        _assert_metric(
            payload,
            metric_name,
            expected_value=value,
            expected_unit=unit,
            expected_error=error,
        )


@pytest.mark.parametrize(
    ("category", "optional_metrics"),
    [
        ("financial_markets", ("sharpe_like",)),
        ("marketing_analytics", ("roas", "ctr", "cac")),
        ("operations", ("productivity", "uptime_rate")),
        ("retail", ("conversion_rate", "gross_margin_rate", "average_ticket")),
        ("healthcare", ("bed_occupancy_rate", "revenue_per_staff_hour")),
    ],
)
def test_optional_fallback_behavior_when_optional_inputs_are_missing(
    category: str,
    optional_metrics: tuple[str, ...],
) -> None:
    payload = _compute_payload(category, extra_inputs={})
    for metric_name in optional_metrics:
        entry = payload[metric_name]
        assert entry["value"] is None
        assert entry["error"] == "insufficient_data"
        assert entry["is_valid"] is False
        assert isinstance(entry["missing_dependencies"], list)


def test_new_categories_are_registered() -> None:
    categories = set(supported_categories())
    assert {
        "financial_markets",
        "marketing_analytics",
        "operations",
        "retail",
        "healthcare",
    }.issubset(categories)


def test_strict_json_output_contract_unchanged() -> None:
    expected_fields = {
        "competitive_analysis",
        "strategic_recommendations",
    }
    assert set(InsightOutput.model_fields.keys()) == expected_fields

    sample = InsightOutput(
        competitive_analysis={
            "summary": "Competitor summary based on benchmark metrics.",
            "market_position": "Peer-relative market position remains stable.",
            "relative_performance": "Growth metric tracks near competitor benchmark median.",
            "key_advantages": ["Stronger ARPU metric versus competitor baseline."],
            "key_vulnerabilities": ["Churn metric trails competitor benchmark."],
            "confidence": 0.9,
        },
        strategic_recommendations={
            "immediate_actions": ["Close competitor churn gap in priority segment."],
            "mid_term_moves": ["Improve growth metric position against competitor benchmark."],
            "defensive_strategies": ["Defend segments where competitor retention strength is highest."],
            "offensive_strategies": ["Exploit competitor weakness in monetization benchmark metrics."],
        },
    )
    payload = sample.model_dump()
    assert set(payload.keys()) == expected_fields
    assert math.isclose(payload["competitive_analysis"]["confidence"], 0.9)
