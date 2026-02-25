from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.category_registry import (
    CategoryRegistryError,
    clear_category_registry_cache,
    primary_metric_for_business_type,
    require_category_pack,
    supported_categories,
)
from app.services.kpi_orchestrator import KPIOrchestrator, _AggregatedInputs


def _write_pack(categories_dir: Path, name: str, payload: dict) -> None:
    categories_dir.mkdir(parents=True, exist_ok=True)
    (categories_dir / f"{name}.yaml").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


@pytest.fixture()
def isolated_registry_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    categories_dir = tmp_path / "categories"
    monkeypatch.setenv("CATEGORY_CONFIG_DIR", str(categories_dir))
    clear_category_registry_cache()
    yield categories_dir
    clear_category_registry_cache()


def test_category_pack_requires_optional_signals(isolated_registry_dir: Path) -> None:
    _write_pack(
        isolated_registry_dir,
        "broken_marketplace",
        {
            "name": "broken_marketplace",
            "metric_aliases": {
                "recurring_revenue": ["marketplace_revenue"],
                "active_customer_count": ["active_buyers"],
                "churned_customer_count": ["lost_buyers"],
            },
            "required_inputs": [
                "recurring_revenue",
                "active_customer_count",
                "churned_customer_count",
            ],
            "deterministic_formulas": {
                "class": "kpi.ecommerce:EcommerceKPIFormula",
                "input_bindings": {
                    "orders": {"source": "agg.subscription_revenues", "default": []},
                    "total_visitors": {"source": "extra.total_visitors", "default": 0},
                    "marketing_spend": {"source": "extra.marketing_spend", "default": 0.0},
                    "new_customers": {"source": "extra.new_customers", "default": 0},
                    "unique_customers": {"source": "agg.active_customers", "default": 0},
                    "previous_revenue": {"source": "agg.previous_revenue", "default": 0.0},
                },
            },
        },
    )

    with pytest.raises(CategoryRegistryError, match="optional_signals"):
        supported_categories()


def test_new_category_pack_is_registry_driven(isolated_registry_dir: Path) -> None:
    _write_pack(
        isolated_registry_dir,
        "marketplace",
        {
            "name": "marketplace",
            "metric_aliases": {
                "recurring_revenue": ["marketplace_revenue", "gmv"],
                "active_customer_count": ["active_buyers"],
                "churned_customer_count": ["lost_buyers"],
            },
            "required_inputs": [
                "recurring_revenue",
                "active_customer_count",
                "churned_customer_count",
            ],
            "category_aliases": ["sales", "marketplace"],
            "deterministic_formulas": {
                "class": "kpi.ecommerce:EcommerceKPIFormula",
                "input_bindings": {
                    "orders": {"source": "agg.subscription_revenues", "default": []},
                    "total_visitors": {"source": "extra.total_visitors", "default": 0},
                    "marketing_spend": {"source": "extra.marketing_spend", "default": 0.0},
                    "new_customers": {"source": "extra.new_customers", "default": 0},
                    "unique_customers": {"source": "agg.active_customers", "default": 0},
                    "previous_revenue": {"source": "agg.previous_revenue", "default": 0.0},
                },
                "validity_rules": {
                    "ltv": {
                        "dependencies": [
                            {"source": "agg.subscription_revenues", "missing_when": "is_empty"},
                            {"source": "agg.active_customers", "missing_when": "is_none"},
                        ]
                    }
                },
            },
            "optional_signals": ["aov", "cac", "purchase_frequency", "ltv"],
            "rate_metrics": ["conversion_rate", "purchase_frequency", "growth_rate"],
        },
    )

    pack = require_category_pack("marketplace")
    assert pack.name == "marketplace"
    assert "marketplace_revenue" in pack.metric_aliases["recurring_revenue"]
    assert primary_metric_for_business_type("marketplace") == "marketplace_revenue"


def test_orchestrator_dispatches_new_category_without_control_flow_changes(
    isolated_registry_dir: Path,
) -> None:
    _write_pack(
        isolated_registry_dir,
        "marketplace",
        {
            "name": "marketplace",
            "metric_aliases": {
                "recurring_revenue": ["marketplace_revenue", "gmv"],
                "active_customer_count": ["active_buyers"],
                "churned_customer_count": ["lost_buyers"],
            },
            "required_inputs": [
                "recurring_revenue",
                "active_customer_count",
                "churned_customer_count",
            ],
            "category_aliases": ["sales", "marketplace"],
            "deterministic_formulas": {
                "class": "kpi.ecommerce:EcommerceKPIFormula",
                "input_bindings": {
                    "orders": {"source": "agg.subscription_revenues", "default": []},
                    "total_visitors": {"source": "extra.total_visitors", "default": 0},
                    "marketing_spend": {"source": "extra.marketing_spend", "default": 0.0},
                    "new_customers": {"source": "extra.new_customers", "default": 0},
                    "unique_customers": {"source": "agg.active_customers", "default": 0},
                    "previous_revenue": {"source": "agg.previous_revenue", "default": 0.0},
                },
                "validity_rules": {
                    "growth_rate": {
                        "dependencies": [
                            {"source": "agg.current_revenue", "missing_when": "is_none"},
                            {"source": "agg.previous_revenue", "missing_when": "is_none"},
                        ]
                    }
                },
            },
            "optional_signals": ["aov", "cac", "purchase_frequency", "ltv"],
            "rate_metrics": ["conversion_rate", "purchase_frequency", "growth_rate"],
        },
    )

    agg_inputs = _AggregatedInputs(
        subscription_revenues=[120.0, 80.0, 100.0],
        active_customers=10,
        lost_customers=2,
        arpu=30.0,
        current_revenue=300.0,
        previous_revenue=200.0,
    )

    metrics, validity = KPIOrchestrator()._compute(
        business_type="marketplace",
        agg_inputs=agg_inputs,
        extra_inputs={
            "total_visitors": 1000,
            "marketing_spend": 500.0,
            "new_customers": 10,
        },
    )

    assert metrics["revenue"] == pytest.approx(300.0)
    assert metrics["growth_rate"] == pytest.approx(0.5)
    assert metrics["conversion_rate"] == pytest.approx(0.003)
    assert validity == {}
