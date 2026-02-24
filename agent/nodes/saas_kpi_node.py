"""
agent/nodes/saas_kpi_node.py

SaaS KPI Fetch Node: retrieves persisted ComputedKPI records and stores only
SaaS-relevant KPI payloads in state.saas_kpi_data.

No risk logic, no forecasting, no math.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agent.nodes.node_result import failed, skipped, success
from agent.state import AgentState
from db.repositories.kpi_repository import KPIRepository
from db.session import SessionLocal

_SAAS_METRICS: frozenset[str] = frozenset({
    "mrr",
    "churn_rate",
    "ltv",
    "growth_rate",
    "arpu",
})


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _query_start() -> datetime:
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _serialize_row(row: Any) -> dict[str, Any]:
    """Convert a ComputedKPI ORM row to a plain JSON-safe dict."""
    computed = row.computed_kpis or {}
    return {
        "entity_name": row.entity_name,
        "period_start": row.period_start.isoformat(),
        "period_end": row.period_end.isoformat(),
        "computed_kpis": {
            k: v
            for k, v in computed.items()
            if k in _SAAS_METRICS
        },
        "created_at": row.created_at.isoformat(),
    }


def saas_kpi_fetch_node(state: AgentState) -> AgentState:
    """
    LangGraph node: fetch SaaS KPI records for the entity and write them to
    state["saas_kpi_data"].
    """
    entity_name: str | None = state.get("entity_name") or None
    period_end = _now_utc()
    period_start = _query_start()

    try:
        with SessionLocal() as session:
            repo = KPIRepository(session)
            rows = repo.get_kpis_by_period(
                period_start=period_start,
                period_end=period_end,
                entity_name=entity_name,
            )
            records: list[dict[str, Any]] = []
            for row in rows:
                payload = _serialize_row(row)
                if payload["computed_kpis"]:
                    records.append(payload)

        payload: dict[str, Any] = {
            "records": records,
            "fetched_for": entity_name,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "metrics": sorted(_SAAS_METRICS),
        }
        if records:
            saas_kpi_data = success(payload)
        else:
            saas_kpi_data = skipped("no_kpi_records", payload)

    except Exception as exc:  # noqa: BLE001
        saas_kpi_data = failed(
            str(exc),
            {
            "fetched_for": entity_name,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "metrics": sorted(_SAAS_METRICS),
            },
        )

    return {**state, "saas_kpi_data": saas_kpi_data}
