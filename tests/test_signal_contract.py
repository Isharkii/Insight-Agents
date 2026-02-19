from agent.signal_normalizer import REQUIRED_SIGNALS, normalize_signals


def test_normalize_signals_contract() -> None:
    kpi_payload = {
        "records": [
            {
                "entity_name": "Acme",
                "period_start": "2026-01-01T00:00:00+00:00",
                "period_end": "2026-01-31T00:00:00+00:00",
                "created_at": "2026-02-01T00:00:00+00:00",
                "computed_kpis": {
                    "revenue_growth_delta": {"value": -0.12, "unit": "rate", "error": None},
                    "churn_delta": {"value": 0.04, "unit": "rate", "error": None},
                    "conversion_delta": {"value": -0.02, "unit": "rate", "error": None},
                },
            },
            {
                "entity_name": "Acme",
                "period_start": "2026-02-01T00:00:00+00:00",
                "period_end": "2026-02-28T00:00:00+00:00",
                "created_at": "2026-03-01T00:00:00+00:00",
                "computed_kpis": {
                    "revenue_growth_delta": {"value": -0.18, "unit": "rate", "error": None},
                    "churn_delta": {"value": 0.07, "unit": "rate", "error": None},
                    "conversion_delta": {"value": -0.03, "unit": "rate", "error": None},
                },
            },
        ]
    }

    forecast_payload = {
        "forecasts": {
            "mrr": {
                "forecast_data": {
                    "metric_name": "mrr",
                    "slope": -0.42,
                    "deviation_percentage": 0.15,
                    "churn_acceleration": 0.08,
                }
            }
        }
    }

    signals = normalize_signals(kpi_payload=kpi_payload, forecast_payload=forecast_payload)

    required = set(REQUIRED_SIGNALS) | {"churn_acceleration"}
    assert required.issubset(signals.keys())

    assert signals["revenue_growth_delta"] == -0.18
    assert signals["churn_delta"] == 0.07
    assert signals["conversion_delta"] == -0.03
    assert signals["slope"] == -0.42
    assert signals["deviation_percentage"] == 0.15
    assert signals["churn_acceleration"] == 0.08

    for key in required:
        assert signals[key] != 0.0
