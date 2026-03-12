from __future__ import annotations

from agent.helpers.kpi_extraction import metric_series_from_kpi_payload, records_from_kpi_payload
from agent.helpers.state_slimming import clear_raw_kpi_cache, slim_kpi_payload
from agent.nodes.growth_engine_node import growth_engine_node
from agent.nodes.node_result import status_of, success
from agent.signal_normalizer import normalize_kpi_signals


def _sample_records() -> list[dict]:
    return [
        {
            "entity_name": "acme",
            "period_start": "2026-01-01T00:00:00+00:00",
            "period_end": "2026-01-31T00:00:00+00:00",
            "computed_kpis": {
                "recurring_revenue": {"value": 100.0},
                "churn_rate": {"value": 0.04},
                "conversion_rate": {"value": 0.02},
            },
            "created_at": "2026-01-31T00:00:00+00:00",
        },
        {
            "entity_name": "acme",
            "period_start": "2026-02-01T00:00:00+00:00",
            "period_end": "2026-02-28T00:00:00+00:00",
            "computed_kpis": {
                "recurring_revenue": {"value": 112.0},
                "churn_rate": {"value": 0.05},
                "conversion_rate": {"value": 0.021},
            },
            "created_at": "2026-02-28T00:00:00+00:00",
        },
    ]


def test_slim_kpi_payload_avoids_raw_records_in_state() -> None:
    clear_raw_kpi_cache()
    payload = slim_kpi_payload(
        _sample_records(),
        fetched_for="acme",
        period_start="2026-01-01T00:00:00+00:00",
        period_end="2026-02-28T00:00:00+00:00",
    )

    assert payload.get("state_mode") == "derived_only"
    assert "records" not in payload
    assert payload.get("record_count") == 2
    assert isinstance(payload.get("record_ref"), str) and payload["record_ref"]
    assert payload.get("metric_series", {}).get("recurring_revenue") == [100.0, 112.0]

    recovered = records_from_kpi_payload(payload)
    assert len(recovered) == 2


def test_signal_normalizer_supports_compact_metric_series_payload() -> None:
    payload = slim_kpi_payload(
        _sample_records(),
        fetched_for="acme",
        period_start="2026-01-01T00:00:00+00:00",
        period_end="2026-02-28T00:00:00+00:00",
    )

    signals = normalize_kpi_signals(payload, strict=True)
    assert "revenue_growth_delta" in signals
    assert "churn_delta" in signals
    assert "conversion_delta" in signals


def test_growth_engine_node_accepts_compact_kpi_payload() -> None:
    payload = slim_kpi_payload(
        _sample_records(),
        fetched_for="acme",
        period_start="2026-01-01T00:00:00+00:00",
        period_end="2026-02-28T00:00:00+00:00",
    )
    state = {
        "business_type": "general_timeseries",
        "entity_name": "acme",
        "kpi_data": success(payload),
    }

    updated = growth_engine_node(state)
    assert status_of(updated.get("growth_data")) == "success"
    growth_payload = updated["growth_data"]["payload"]
    assert growth_payload.get("primary_metric")

    series = metric_series_from_kpi_payload(payload)
    assert "recurring_revenue" in series

