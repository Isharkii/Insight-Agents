from __future__ import annotations

from agent.graph import role_analytics_node
from agent.helpers.statistical_context import build_statistical_context
from agent.nodes.node_result import status_of
from app.services.statistics.anomaly import detect_iqr_anomalies, iqr_bounds
from app.services.statistics.normalization import (
    metric_statistics_config,
    rolling_mean,
    rolling_median,
    zscore_normalize,
)


def test_zscore_normalization_with_clipping_is_deterministic() -> None:
    series = [1.0, 2.0, 3.0, 100.0]
    first = zscore_normalize(series, clip_abs=1.0)
    second = zscore_normalize(series, clip_abs=1.0)

    assert first == second
    assert max(first) <= 1.0
    assert min(first) >= -1.0
    assert first[-1] == 1.0


def test_rolling_mean_and_median_smoothing() -> None:
    series = [1.0, 100.0, 2.0, 3.0]
    smoothed_mean = rolling_mean(series, window=3)
    smoothed_median = rolling_median(series, window=3)

    assert smoothed_mean == [1.0, 50.5, 34.333333, 35.0]
    assert smoothed_median == [1.0, 50.5, 2.0, 3.0]


def test_iqr_anomaly_bounds_and_flags() -> None:
    series = [10.0, 11.0, 12.0, 13.0, 200.0]
    bounds = iqr_bounds(series, multiplier=1.5)
    anomalies = detect_iqr_anomalies(series, multiplier=1.5)

    assert bounds["lower_bound"] is not None
    assert bounds["upper_bound"] is not None
    assert anomalies["status"] == "ok"
    assert anomalies["anomaly_indexes"] == [4]
    assert anomalies["anomaly_values"] == [200.0]
    assert anomalies["anomaly_flags"] == [False, False, False, False, True]


def test_metric_statistics_config_uses_per_metric_overrides() -> None:
    churn_cfg = metric_statistics_config("churn_rate")
    default_cfg = metric_statistics_config("unconfigured_metric")

    assert churn_cfg.smoothing_method == "median"
    assert churn_cfg.smoothing_window == 4
    assert churn_cfg.anomaly_iqr_multiplier == 1.5
    assert default_cfg.smoothing_window == 3
    assert default_cfg.smoothing_method == "mean"


def test_role_analytics_payload_includes_statistical_context(monkeypatch) -> None:
    def _fake_fetch_canonical_dimension_rows(**_: object) -> list[dict]:
        return [
            {
                "role": "Team Alpha",
                "team": "Team Alpha",
                "channel": "Paid",
                "region": "US",
                "product_line": "Core",
                "source_type": "csv",
                "metric_name": "revenue",
                "metric_value": 120.0,
                "metadata_json": {},
            },
            {
                "role": "Team Beta",
                "team": "Team Beta",
                "channel": "Organic",
                "region": "EU",
                "product_line": "Expansion",
                "source_type": "csv",
                "metric_name": "revenue",
                "metric_value": 80.0,
                "metadata_json": {},
            },
        ]

    monkeypatch.setattr(
        "agent.graph._fetch_canonical_dimension_rows",
        _fake_fetch_canonical_dimension_rows,
    )
    monkeypatch.setattr(
        "agent.graph._fetch_macro_context_rows",
        lambda **_: ([], []),
    )

    # Pre-compute the statistical context that the refactored node reads
    # from the upstream multivariate_scenario_data envelope.
    metric_series = {"timeseries_value": [100.0, 105.0, 160.0, 110.0]}
    stat_ctx = build_statistical_context(metric_series)

    upstream_multivariate = {
        "status": "success",
        "payload": {
            "statistical_context": stat_ctx,
            "multivariate_context": {},
            "scenario_simulation": {},
        },
        "confidence_score": 0.8,
    }

    state = {
        "business_type": "general_timeseries",
        "entity_name": "Acme",
        "multivariate_scenario_data": upstream_multivariate,
        "kpi_data": {
            "status": "success",
            "payload": {
                "fetched_for": "Acme",
                "period_start": "2026-01-01T00:00:00+00:00",
                "period_end": "2026-04-01T00:00:00+00:00",
                "records": [
                    {
                        "period_end": "2026-01-01T00:00:00+00:00",
                        "computed_kpis": {"timeseries_value": {"value": 100.0}},
                    },
                    {
                        "period_end": "2026-02-01T00:00:00+00:00",
                        "computed_kpis": {"timeseries_value": {"value": 105.0}},
                    },
                    {
                        "period_end": "2026-03-01T00:00:00+00:00",
                        "computed_kpis": {"timeseries_value": {"value": 160.0}},
                    },
                    {
                        "period_end": "2026-04-01T00:00:00+00:00",
                        "computed_kpis": {"timeseries_value": {"value": 110.0}},
                    },
                ],
            },
        },
    }

    updated = role_analytics_node(state)
    segmentation = updated["segmentation"]
    assert status_of(segmentation) == "success"
    payload = segmentation["payload"]
    assert "statistical_context" in payload

    stat_result = payload["statistical_context"]
    assert "metrics" in stat_result
    assert "timeseries_value" in stat_result["metrics"]

    metric_payload = stat_result["metrics"]["timeseries_value"]
    assert "zscore" in metric_payload
    assert "smoothing" in metric_payload
    assert "anomaly" in metric_payload
    assert metric_payload["anomaly"]["status"] == "ok"

