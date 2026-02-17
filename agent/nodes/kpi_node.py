"""
agent/nodes/kpi_node.py

KPI Fetch Node: loads the latest ComputedKPI records for the entity
named in state and writes them into state.kpi_data.

No KPI calculation, no forecasting, no risk logic, no LLM.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from agent.state import AgentState
from db.repositories.kpi_repository import KPIRepository
from db.session import SessionLocal

# Default look-back window when no explicit period is in state.
_DEFAULT_WINDOW_DAYS: int = 30


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _serialize_row(row: Any) -> dict[str, Any]:
    """Convert a ComputedKPI ORM row to a plain JSON-safe dict."""
    return {
        "entity_name": row.entity_name,
        "period_start": row.period_start.isoformat(),
        "period_end": row.period_end.isoformat(),
        "computed_kpis": row.computed_kpis,
        "created_at": row.created_at.isoformat(),
    }


def kpi_fetch_node(state: AgentState) -> AgentState:
    """
    LangGraph node: fetch ComputedKPI records from the repository.

    Reads:
        state["entity_name"]  — scopes the query; empty → fetch all entities.

    Writes:
        state["kpi_data"] — dict with keys:
            "records"      : list of serialised ComputedKPI rows
            "fetched_for"  : entity_name used for the query (may be None)
            "period_start" : ISO string of window start
            "period_end"   : ISO string of window end
            "error"        : present only on failure
    """
    entity_name: str | None = state.get("entity_name") or None
    period_end: datetime = _now_utc()
    period_start: datetime = period_end - timedelta(days=_DEFAULT_WINDOW_DAYS)

    try:
        with SessionLocal() as session:
            repo = KPIRepository(session)
            rows = repo.get_kpis_by_period(
                period_start=period_start,
                period_end=period_end,
                entity_name=entity_name,
            )
            records = [_serialize_row(r) for r in rows]

        kpi_data: dict[str, Any] = {
            "records": records,
            "fetched_for": entity_name,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
        }

    except Exception as exc:  # noqa: BLE001
        kpi_data = {
            "records": [],
            "fetched_for": entity_name,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "error": str(exc),
        }

    return {**state, "kpi_data": kpi_data}
