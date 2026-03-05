from __future__ import annotations

from agent.nodes.node_result import success
from agent.signal_integrity import UnifiedSignalIntegrity


def test_unified_signal_integrity_applies_kpi_gate() -> None:
    state = {
        "business_type": "saas",
        "saas_kpi_data": success(
            {
                "metrics": ["mrr", "churn_rate"],
                "records": [
                    {
                        "computed_kpis": {
                            "mrr": {"value": 100.0, "source": "formula"},
                        },
                    }
                ],
            }
        ),
    }

    result = UnifiedSignalIntegrity.compute(state)

    assert result["layers"]["kpi"]["score"] < 0.3
    assert result["kpi_gate_passed"] is False
    assert result["overall_score"] == 0.0


def test_unified_signal_integrity_forecast_insufficient_data_scores_zero() -> None:
    state = {
        "business_type": "saas",
        "saas_kpi_data": success(
            {
                "metrics": ["mrr", "churn_rate"],
                "records": [
                    {
                        "computed_kpis": {
                            "mrr": {"value": 100.0, "source": "formula"},
                            "churn_rate": {"value": 0.05, "source": "formula"},
                        },
                    },
                    {
                        "computed_kpis": {
                            "mrr": {"value": 110.0, "source": "formula"},
                            "churn_rate": {"value": 0.04, "source": "formula"},
                        },
                    },
                    {
                        "computed_kpis": {
                            "mrr": {"value": 120.0, "source": "formula"},
                            "churn_rate": {"value": 0.03, "source": "formula"},
                        },
                    },
                ],
            }
        ),
        "forecast_data": {"status": "insufficient_data", "payload": {}},
    }

    result = UnifiedSignalIntegrity.compute(state)
    assert result["layers"]["forecast"]["score"] == 0.0
    assert result["kpi_gate_passed"] is True


def test_unified_signal_integrity_competitive_external_fetch() -> None:
    state = {
        "business_type": "saas",
        "saas_kpi_data": success(
            {
                "metrics": ["mrr", "churn_rate"],
                "records": [
                    {
                        "computed_kpis": {
                            "mrr": {"value": 100.0, "source": "formula"},
                            "churn_rate": {"value": 0.05, "source": "formula"},
                        },
                    },
                    {
                        "computed_kpis": {
                            "mrr": {"value": 110.0, "source": "formula"},
                            "churn_rate": {"value": 0.04, "source": "formula"},
                        },
                    },
                    {
                        "computed_kpis": {
                            "mrr": {"value": 120.0, "source": "formula"},
                            "churn_rate": {"value": 0.03, "source": "formula"},
                        },
                    },
                ],
            }
        ),
        "competitive_context": {
            "source": "external_fetch",
            "peer_count": 3,
            "metrics": ["mrr", "churn_rate"],
            "benchmark_rows_count": 12,
        },
    }

    result = UnifiedSignalIntegrity.compute(state)
    layer = result["layers"]["competitive"]
    assert layer["source_reliability"] == 0.7
    assert layer["score"] > 0.0
