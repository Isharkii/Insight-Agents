from __future__ import annotations

from agent.graph import role_analytics_node
from agent.nodes.node_result import status_of
from agent.nodes.prioritization_node import prioritization_node
from agent.nodes.risk_node import risk_node
from app.services.statistics.multivariate import compute_multivariate_context
from app.services.statistics.scenario_simulator import simulate_deterministic_scenarios


def test_multivariate_correlation_significance_and_variance_decomposition() -> None:
    metric_series = {
        "revenue": [10, 20, 30, 40, 50, 60, 70],
        "active_customer_count": [5, 10, 15, 20, 25, 30, 35],
        "churn_rate": [0.20, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13],
    }
    segment_rows = [
        {"segment": "SMB", "metric_value": 140.0},
        {"segment": "Enterprise", "metric_value": 60.0},
    ]

    context = compute_multivariate_context(
        metric_series,
        segment_rows=segment_rows,
        preferred_metric_candidates=("revenue",),
    )

    assert context["status"] in {"success", "partial"}
    corr = context["correlation"]
    assert corr["matrix"]["revenue"]["active_customer_count"] is not None
    assert corr["filtered_matrix"]["revenue"]["active_customer_count"] is not None

    variance_share = context["variance_decomposition"]["variance_share"]
    share_sum = sum(float(v) for v in variance_share.values())
    assert abs(share_sum - 1.0) < 1e-6

    top_segment = context["segment_contribution"]["top_segment"]
    assert top_segment["segment"] == "SMB"
    assert abs(float(top_segment["contribution_share"]) - 0.7) < 1e-6


def test_scenario_simulator_outputs_deterministic_best_base_worst_and_metadata() -> None:
    metric_series = {
        "revenue": [100.0, 105.0, 110.0, 118.0, 123.0],
    }
    growth_context = {
        "primary_metric": "revenue",
        "confidence_score": 0.9,
        "primary_horizons": {
            "short_growth": 0.06,
            "mid_growth": 0.05,
            "long_growth": 0.04,
            "trend_acceleration": 0.01,
            "insufficient_history": {
                "short": False,
                "mid": False,
                "long": False,
                "cagr": False,
                "acceleration": False,
            },
        },
    }
    statistical_context = {
        "confidence_score": 0.85,
        "anomaly_summary": {"total_anomaly_points": 2},
    }
    multivariate_context = {
        "confidence_score": 0.8,
        "correlation": {"total_pairs": 6, "significant_pairs": 4},
    }

    output = simulate_deterministic_scenarios(
        metric_series,
        growth_context=growth_context,
        statistical_context=statistical_context,
        multivariate_context=multivariate_context,
        preferred_metric_candidates=("revenue",),
    )

    scenarios = output["scenarios"]
    assert set(scenarios.keys()) == {"best", "base", "worst"}
    assert scenarios["best"]["projected_value"] > scenarios["base"]["projected_value"]
    assert scenarios["worst"]["projected_value"] < scenarios["base"]["projected_value"]

    metadata = output["metadata"]
    assert "assumptions" in metadata
    assert "confidence_impact" in metadata
    assert "shocks" in metadata["assumptions"]
    assert "confidence_impact" in scenarios["worst"]["assumptions"]


def test_role_analytics_payload_includes_multivariate_and_scenarios(monkeypatch) -> None:
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
    monkeypatch.setattr("agent.graph._fetch_macro_context_rows", lambda **_: ([], []))

    # Supply upstream multivariate_scenario_data that role_analytics reads
    # (the refactored node no longer re-computes missing upstream signals).
    upstream_multivariate = {
        "status": "success",
        "payload": {
            "statistical_context": {"confidence_score": 0.85, "anomaly_summary": {}},
            "multivariate_context": {"confidence_score": 0.8, "correlation": {}},
            "scenario_simulation": {
                "status": "success",
                "base_confidence": 0.75,
                "scenarios": {
                    "best": {"projected_growth": 0.05},
                    "base": {"projected_growth": 0.02},
                    "worst": {"projected_growth": -0.04},
                },
            },
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
                "period_end": "2026-05-01T00:00:00+00:00",
                "records": [
                    {
                        "period_end": "2026-01-01T00:00:00+00:00",
                        "computed_kpis": {
                            "timeseries_value": {"value": 100.0},
                            "active_customer_count": {"value": 50.0},
                        },
                    },
                    {
                        "period_end": "2026-02-01T00:00:00+00:00",
                        "computed_kpis": {
                            "timeseries_value": {"value": 110.0},
                            "active_customer_count": {"value": 56.0},
                        },
                    },
                    {
                        "period_end": "2026-03-01T00:00:00+00:00",
                        "computed_kpis": {
                            "timeseries_value": {"value": 118.0},
                            "active_customer_count": {"value": 60.0},
                        },
                    },
                    {
                        "period_end": "2026-04-01T00:00:00+00:00",
                        "computed_kpis": {
                            "timeseries_value": {"value": 115.0},
                            "active_customer_count": {"value": 58.0},
                        },
                    },
                    {
                        "period_end": "2026-05-01T00:00:00+00:00",
                        "computed_kpis": {
                            "timeseries_value": {"value": 122.0},
                            "active_customer_count": {"value": 62.0},
                        },
                    },
                ],
            },
        },
    }

    updated = role_analytics_node(state)
    segmentation = updated["segmentation"]
    assert status_of(segmentation) == "success"
    payload = segmentation["payload"]
    assert "multivariate_context" in payload
    assert "scenario_simulation" in payload
    assert "statistical_context" in payload


def test_risk_and_prioritization_expose_scenario_metadata(monkeypatch) -> None:
    class _DummySession:
        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> bool:
            return False

        def commit(self) -> None:
            return None

    class _FakeRiskOrchestrator:
        def __init__(self, _session: object) -> None:
            self._session = _session

        def generate_risk_score(self, entity_name: str, kpi_data: dict, forecast_data: dict) -> dict:
            assert forecast_data["forecast_available"] is True
            assert float(forecast_data["slope"]) == -0.12
            return {
                "entity_name": entity_name,
                "risk_score": 72,
                "risk_level": "high",
            }

    monkeypatch.setattr("agent.nodes.risk_node.SessionLocal", lambda: _DummySession())
    monkeypatch.setattr("agent.nodes.risk_node.RiskOrchestrator", _FakeRiskOrchestrator)

    segmentation_payload = {
        "scenario_simulation": {
            "status": "success",
            "base_confidence": 0.82,
            "insufficient_history": {"short": False},
            "warnings": [],
            "scenarios": {
                "best": {"projected_growth": 0.03, "assumptions": {"confidence_impact": 0.05}},
                "base": {"projected_growth": -0.02, "assumptions": {"confidence_impact": 0.0}},
                "worst": {"projected_growth": -0.12, "assumptions": {"confidence_impact": -0.08}},
            },
            "metadata": {
                "assumptions": {"projection_periods": 1},
                "confidence_impact": {"confidence_delta_worst_vs_base": -0.08},
            },
        }
    }

    risk_state = {
        "business_type": "general_timeseries",
        "entity_name": "Acme",
        "kpi_data": {
            "status": "success",
            "payload": {
                "records": [
                    {
                        "period_end": "2026-01-31T00:00:00+00:00",
                        "created_at": "2026-01-31T00:00:00+00:00",
                        "computed_kpis": {
                            "revenue": {"value": 100.0},
                            "churn_rate": {"value": 0.05},
                            "conversion_rate": {"value": 0.02},
                        },
                    },
                    {
                        "period_end": "2026-02-28T00:00:00+00:00",
                        "created_at": "2026-02-28T00:00:00+00:00",
                        "computed_kpis": {
                            "revenue": {"value": 95.0},
                            "churn_rate": {"value": 0.07},
                            "conversion_rate": {"value": 0.018},
                        },
                    },
                ]
            },
        },
        "forecast_data": None,
        "segmentation": {"status": "success", "payload": segmentation_payload},
    }
    risk_updated = risk_node(risk_state)
    risk_payload = risk_updated["risk_data"]["payload"]
    assert risk_payload["scenario_signals_used"] is True
    assert risk_payload["scenario_signal_status"] == "success"
    assert risk_payload["scenario_worst_growth"] == -0.12
    assert "scenario_assumptions" in risk_payload

    prio_state = {
        "risk_data": {"status": "success", "payload": {"risk_score": 72.0, "risk_level": "high"}},
        "root_cause": {"status": "success", "payload": {}},
        "segmentation": {"status": "success", "payload": segmentation_payload},
    }
    prio_updated = prioritization_node(prio_state)
    prio_payload = prio_updated["prioritization"]
    assert prio_payload["scenario_signal_used"] is True
    assert prio_payload["scenario_worst_growth"] == -0.12
    assert "scenario_assumptions" in prio_payload

