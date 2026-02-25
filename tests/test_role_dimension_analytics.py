from __future__ import annotations

from agent.graph import role_analytics_node
from agent.nodes.node_result import status_of
from app.services.role_dimension_analytics import (
    SUPPORTED_ROLE_DIMENSIONS,
    build_role_dimension_summary,
)


def test_build_role_dimension_summary_returns_unified_schema() -> None:
    rows = [
        {
            "role": "Team Alpha",
            "team": "Team Alpha",
            "channel": "Paid",
            "region": "US",
            "product_line": "Core",
            "metric_value": 120.0,
            "metadata_json": {},
        },
        {
            "role": "Team Beta",
            "team": "Team Beta",
            "channel": "Organic",
            "region": "EU",
            "product_line": "Core",
            "metric_value": 80.0,
            "metadata_json": {},
        },
        {
            "role": "Team Alpha",
            "team": "Team Alpha",
            "channel": "Paid",
            "region": "US",
            "product_line": "Expansion",
            "metric_value": 50.0,
            "metadata_json": {},
        },
    ]

    summary = build_role_dimension_summary(rows, top_n=2)
    assert summary["dimensions"] == list(SUPPORTED_ROLE_DIMENSIONS)
    assert "top_contributors" in summary
    assert "laggards" in summary
    assert "dependency_concentration" in summary
    assert "by_dimension" in summary
    assert summary["records_used"] == 3
    assert len(summary["top_contributors"]) <= 2

    team = summary["by_dimension"]["team"]
    assert team["contributors"][0]["name"] == "Team Alpha"
    assert team["contributors"][0]["contribution_value"] == 170.0


def test_role_analytics_node_exposes_unified_dimension_payload(monkeypatch) -> None:
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

    state = {
        "business_type": "general_timeseries",
        "entity_name": "Acme",
        "kpi_data": {
            "status": "success",
            "payload": {
                "fetched_for": "Acme",
                "period_start": "2026-01-01T00:00:00+00:00",
                "period_end": "2026-01-31T00:00:00+00:00",
                "records": [{"computed_kpis": {"timeseries_value": {"value": 200.0}}}],
            },
        },
    }

    updated = role_analytics_node(state)
    segmentation = updated["segmentation"]
    assert status_of(segmentation) == "success"
    payload = segmentation["payload"]
    assert "top_contributors" in payload
    assert "laggards" in payload
    assert "dependency_concentration" in payload
    assert "by_dimension" in payload
    assert payload["by_dimension"]["team"]["top_contributors"][0]["name"] == "Team Alpha"
