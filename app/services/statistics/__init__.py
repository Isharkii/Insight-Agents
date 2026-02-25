from app.services.statistics.anomaly import detect_iqr_anomalies, iqr_bounds
from app.services.statistics.growth_engine import (
    GrowthEngineConfig,
    compute_growth_context,
    compute_growth_signals,
    metric_growth_config,
)
from app.services.statistics.multivariate import (
    MultivariateConfig,
    compute_multivariate_context,
)
from app.services.statistics.inflation_adjustment import (
    align_cpi_to_kpi_periods,
    build_inflation_adjusted_series,
    compound_inflation_rates,
    compute_cpi_inflation_rate,
    compute_inflation_adjusted_revenue,
    compute_real_growth_rate as compute_real_growth_rate_with_cpi,
)
from app.services.statistics.macro_normalizer import (
    align_macro_series_to_periods,
    compute_growth_from_levels,
    compute_growth_vs_gdp_delta,
    compute_rate_adjusted_efficiency,
    compute_real_growth_rate,
    normalize_business_metrics_for_macro,
    normalize_rate_series,
)
from app.services.statistics.normalization import (
    MetricStatisticsConfig,
    metric_statistics_config,
    rolling_mean,
    rolling_median,
    zscore_normalize,
)
from app.services.statistics.scenario_simulator import (
    ScenarioConfig,
    ScenarioShock,
    simulate_deterministic_scenarios,
)

__all__ = [
    "MetricStatisticsConfig",
    "metric_statistics_config",
    "zscore_normalize",
    "rolling_mean",
    "rolling_median",
    "iqr_bounds",
    "detect_iqr_anomalies",
    "GrowthEngineConfig",
    "metric_growth_config",
    "compute_growth_signals",
    "compute_growth_context",
    "MultivariateConfig",
    "compute_multivariate_context",
    "align_cpi_to_kpi_periods",
    "compute_cpi_inflation_rate",
    "compute_real_growth_rate_with_cpi",
    "compute_inflation_adjusted_revenue",
    "compound_inflation_rates",
    "build_inflation_adjusted_series",
    "compute_growth_from_levels",
    "normalize_rate_series",
    "align_macro_series_to_periods",
    "compute_real_growth_rate",
    "compute_growth_vs_gdp_delta",
    "compute_rate_adjusted_efficiency",
    "normalize_business_metrics_for_macro",
    "ScenarioConfig",
    "ScenarioShock",
    "simulate_deterministic_scenarios",
]
