from __future__ import annotations

from datetime import datetime, timezone

from agent.graph import (
    _filter_peers_by_size_band,
    _infer_size_band_from_metadata_rows,
    role_analytics_node,
)
from agent.nodes.node_result import status_of
from app.services.macro_context_service import build_macro_context


def _kpi_payload_with_monthly_revenue(points: int = 24) -> dict:
    records: list[dict] = []
    for i in range(points):
        year = 2024 + (i // 12)
        month = (i % 12) + 1
        value = 100.0 + float(i)
        if month in (11, 12):
            value += 35.0
        if month in (6, 7):
            value -= 20.0
        ts = datetime(year=year, month=month, day=28, tzinfo=timezone.utc)
        records.append(
            {
                "period_end": ts.isoformat(),
                "created_at": ts.isoformat(),
                "computed_kpis": {
                    "timeseries_value": {"value": value},
                },
            }
        )
    return {
        "fetched_for": "Acme",
        "records": records,
    }


def test_build_macro_context_with_external_series_available() -> None:
    payload = _kpi_payload_with_monthly_revenue(points=24)
    inflation_rows = [
        {
            "entity_name": "WLD",
            "category": "macro",
            "source_type": "api",
            "metric_name": "cpi",
            "metric_value": 100.0,
            "timestamp": "2025-01-01T00:00:00+00:00",
            "metadata_json": {"provider": "world_bank"},
        },
        {
            "entity_name": "WLD",
            "category": "macro",
            "source_type": "api",
            "metric_name": "cpi",
            "metric_value": 105.0,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "metadata_json": {"provider": "world_bank"},
        },
    ]
    benchmark_rows = [
        {
            "entity_name": "Acme",
            "category": "sales",
            "source_type": "csv",
            "metric_name": "industry_benchmark_index",
            "metric_value": 100.0,
            "timestamp": "2025-01-01T00:00:00+00:00",
            "metadata_json": {"source": "analyst_pack_v1"},
        },
        {
            "entity_name": "Acme",
            "category": "sales",
            "source_type": "csv",
            "metric_name": "industry_benchmark_index",
            "metric_value": 104.0,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "metadata_json": {"source": "analyst_pack_v1"},
        },
    ]

    context = build_macro_context(
        kpi_payload=payload,
        inflation_rows=inflation_rows,
        benchmark_rows=benchmark_rows,
        metric_candidates=("timeseries_value",),
    )

    assert context["signals"]["seasonality"]["status"] == "available"
    assert context["signals"]["inflation"]["status"] == "available"
    assert context["signals"]["industry_benchmark"]["status"] == "available"
    assert context["real_growth_adjustment"]["status"] == "available"
    assert context["benchmark_comparison"]["status"] == "available"

    benchmark_sources = context["reproducibility"]["benchmark_sources"]
    assert isinstance(benchmark_sources, list)
    assert benchmark_sources
    assert benchmark_sources[-1]["timestamp"] == "2026-01-01T00:00:00+00:00"
    assert benchmark_sources[-1]["metadata"]["source"] == "analyst_pack_v1"


def test_build_macro_context_marks_optional_signals_when_missing() -> None:
    payload = _kpi_payload_with_monthly_revenue(points=12)
    context = build_macro_context(
        kpi_payload=payload,
        inflation_rows=[],
        benchmark_rows=[],
        metric_candidates=("timeseries_value",),
    )

    assert context["status"] == "partial"
    assert context["signals"]["inflation"]["status"] == "missing_optional"
    assert context["signals"]["industry_benchmark"]["status"] == "missing_optional"
    assert context["real_growth_adjustment"]["status"] == "missing_optional"
    assert context["benchmark_comparison"]["status"] == "missing_optional"


def test_role_analytics_payload_includes_macro_context(monkeypatch) -> None:
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

    def _fake_fetch_macro_context_rows(**_: object) -> tuple[list[dict], list[dict]]:
        inflation_rows = [
            {
                "entity_name": "WLD",
                "category": "macro",
                "source_type": "api",
                "metric_name": "inflation_rate",
                "metric_value": 0.03,
                "timestamp": "2026-01-01T00:00:00+00:00",
                "metadata_json": {"provider": "mock_macro_source"},
            },
        ]
        benchmark_rows = [
            {
                "entity_name": "Acme",
                "category": "sales",
                "source_type": "csv",
                "metric_name": "industry_benchmark",
                "metric_value": 0.02,
                "timestamp": "2026-01-01T00:00:00+00:00",
                "metadata_json": {"source": "mock_benchmark_pack"},
            },
        ]
        return inflation_rows, benchmark_rows

    monkeypatch.setattr(
        "agent.graph._fetch_canonical_dimension_rows",
        _fake_fetch_canonical_dimension_rows,
    )
    monkeypatch.setattr(
        "agent.graph._fetch_macro_context_rows",
        _fake_fetch_macro_context_rows,
    )

    state = {
        "business_type": "general_timeseries",
        "entity_name": "Acme",
        "kpi_data": {
            "status": "success",
            "payload": {
                "fetched_for": "Acme",
                "period_start": "2026-01-01T00:00:00+00:00",
                "period_end": "2026-02-01T00:00:00+00:00",
                "records": [
                    {
                        "period_end": "2026-01-01T00:00:00+00:00",
                        "computed_kpis": {"timeseries_value": {"value": 200.0}},
                    },
                    {
                        "period_end": "2026-02-01T00:00:00+00:00",
                        "computed_kpis": {"timeseries_value": {"value": 210.0}},
                    },
                ],
            },
        },
    }

    updated = role_analytics_node(state)
    segmentation = updated["segmentation"]
    assert status_of(segmentation) == "success"
    payload = segmentation["payload"]
    assert "macro_context" in payload
    macro_context = payload["macro_context"]
    assert macro_context["signals"]["inflation"]["status"] == "available"
    assert macro_context["signals"]["industry_benchmark"]["status"] == "available"
    assert macro_context["reproducibility"]["benchmark_sources"][0]["timestamp"] == (
        "2026-01-01T00:00:00+00:00"
    )


def test_infer_size_band_from_metadata_rows() -> None:
    rows = [
        ({"random_key": "x"},),
        ({"company_size_band": " Mid_Market "},),
    ]
    assert _infer_size_band_from_metadata_rows(rows) == "mid_market"


def test_filter_peers_by_size_band_prefers_matches() -> None:
    benchmark_rows = [
        {"entity_name": "PeerA", "metadata_json": {"size_band": "mid_market"}},
        {"entity_name": "PeerB", "metadata_json": {"size_band": "enterprise"}},
    ]
    filtered = _filter_peers_by_size_band(
        benchmark_rows=benchmark_rows,
        client_size_band="mid_market",
    )
    assert len(filtered) == 1
    assert filtered[0]["entity_name"] == "PeerA"


def test_filter_peers_by_size_band_falls_back_when_no_matches() -> None:
    benchmark_rows = [
        {"entity_name": "PeerA", "metadata_json": {"size_band": "enterprise"}},
        {"entity_name": "PeerB", "metadata_json": {"size_band": "enterprise"}},
    ]
    filtered = _filter_peers_by_size_band(
        benchmark_rows=benchmark_rows,
        client_size_band="mid_market",
    )
    assert filtered == benchmark_rows
