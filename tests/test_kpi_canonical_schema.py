from app.services.kpi_canonical_schema import (
    category_aliases_for_business_type,
    infer_analytics_strategy_from_categories,
    metric_aliases_for_business_type,
)


def test_metric_aliases_for_saas_include_legacy_csv_names() -> None:
    aliases = metric_aliases_for_business_type("saas")
    assert "mrr" in aliases["recurring_revenue"]
    assert "active_customers" in aliases["active_customer_count"]
    assert "churned_subscriptions" in aliases["churned_customer_count"]


def test_category_aliases_for_saas_include_sales_and_saas() -> None:
    categories = category_aliases_for_business_type("saas")
    assert "sales" in categories
    assert "saas" in categories


def test_infer_analytics_strategy_from_categories_single_match() -> None:
    inferred = infer_analytics_strategy_from_categories(["macro", "saas"])
    assert inferred == "saas"


def test_infer_analytics_strategy_from_categories_ambiguous_returns_none() -> None:
    inferred = infer_analytics_strategy_from_categories(["saas", "ecommerce"])
    assert inferred is None
