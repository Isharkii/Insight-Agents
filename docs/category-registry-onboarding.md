# Category Registry Onboarding

This project loads KPI category behavior from YAML packs in `config/categories/`.
To add a new category, add one new pack file. Do not edit orchestrator control flow.

## Required Pack Fields

Each category pack must define:

- `name`
- `metric_aliases`
- `required_inputs`
- `deterministic_formulas`
- `optional_signals`

### `deterministic_formulas` required keys

- `class`: Python import path in `module:ClassName` format (must implement `BaseKPIFormula`)
- `input_bindings`: formula input mapping to sources (`agg.<field>` or `extra.<field>`)

### Optional pack fields

- `category_aliases`
- `rate_metrics`
- `deterministic_formulas.validity_rules`

## Template

```yaml
name: marketplace
metric_aliases:
  recurring_revenue: [marketplace_revenue, gmv]
  active_customer_count: [active_buyers]
  churned_customer_count: [lost_buyers]
required_inputs: [recurring_revenue, active_customer_count, churned_customer_count]
category_aliases: [sales, marketplace]
deterministic_formulas:
  class: kpi.ecommerce:EcommerceKPIFormula
  input_bindings:
    orders: {source: agg.subscription_revenues, default: []}
    total_visitors: {source: extra.total_visitors, default: 0}
    marketing_spend: {source: extra.marketing_spend, default: 0.0}
    new_customers: {source: extra.new_customers, default: 0}
    unique_customers: {source: agg.active_customers, default: 0}
    previous_revenue: {source: agg.previous_revenue, default: 0.0}
optional_signals: [aov, cac, purchase_frequency, ltv]
rate_metrics: [conversion_rate, purchase_frequency, growth_rate]
```

## Dispatch Behavior

- `KPIOrchestrator` resolves `business_type` through `app/services/category_registry.py`.
- Formula class and input binding dispatch are pack-driven.
- Forecast primary metric selection is also pack-driven (`metric_aliases.recurring_revenue`).

## Verification

Run:

```powershell
pytest tests/test_category_registry_onboarding.py
```

These tests validate that a new category pack can be added and dispatched without changing orchestrator branches.
