"""
agent/nodes/agency_kpi_node.py

Agency KPI Fetch Node: retrieves persisted ComputedKPI records and stores only
agency-relevant KPI payloads in state.agency_kpi_data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agent.state import AgentState
from db.repositories.kpi_repository import KPIRepository
from db.session import SessionLocal

_AGENCY_METRICS: frozenset[str] = frozenset({
    "retainer_revenue",
    "project_revenue",
    "total_revenue",
    "client_churn",
    "utilization_rate",
    "revenue_per_employee",
    "client_ltv",
})


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _query_start() -> datetime:
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _serialize_row(row: Any) -> dict[str, Any]:
    computed = row.computed_kpis or {}
    return {
        "entity_name": row.entity_name,
        "period_start": row.period_start.isoformat(),
        "period_end": row.period_end.isoformat(),
        "computed_kpis": {
            key: value
            for key, value in computed.items()
            if key in _AGENCY_METRICS
        },
        "created_at": row.created_at.isoformat(),
    }


def agency_kpi_fetch_node(state: AgentState) -> AgentState:
    """
    LangGraph node: fetch agency KPI records and store them in
    state["agency_kpi_data"].
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

        agency_kpi_data: dict[str, Any] = {
            "records": records,
            "fetched_for": entity_name,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "metrics": sorted(_AGENCY_METRICS),
        }

    except Exception as exc:  # noqa: BLE001
        agency_kpi_data = {
            "records": [],
            "fetched_for": entity_name,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "metrics": sorted(_AGENCY_METRICS),
            "error": str(exc),
        }

    return {**state, "agency_kpi_data": agency_kpi_data}
