"""
agent/nodes/saas_kpi_node.py

SaaS KPI Fetch Node: retrieves the latest ComputedKPI records for a SaaS
entity and stores only the SaaS-relevant metrics in state.saas_kpi_data.

No risk logic, no forecasting, no math.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from agent.state import AgentState
from db.repositories.kpi_repository import KPIRepository
from db.session import SessionLocal

_WINDOW_DAYS: int = 30

_SAAS_METRICS: frozenset[str] = frozenset({
    "mrr",
    "churn_rate",
    "ltv",
    "growth_rate",
    "arpu",
})


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _serialize_row(row: Any) -> dict[str, Any]:
    """Convert a ComputedKPI ORM row to a plain JSON-safe dict."""
    return {
        "entity_name": row.entity_name,
        "period_start": row.period_start.isoformat(),
        "period_end": row.period_end.isoformat(),
        "computed_kpis": {
            k: v
            for k, v in row.computed_kpis.items()
            if k in _SAAS_METRICS
        },
        "created_at": row.created_at.isoformat(),
    }


def saas_kpi_fetch_node(state: AgentState) -> AgentState:
    """
    LangGraph node: fetch SaaS KPI records for the entity.

    Reads:
        state["entity_name"] — entity whose KPIs are fetched.

    Writes:
        state["saas_kpi_data"] — dict with keys:
            "records"      : list of serialised ComputedKPI rows
                             (computed_kpis filtered to SaaS metrics only)
            "fetched_for"  : entity_name used for the query
            "period_start" : ISO string of window start
            "period_end"   : ISO string of window end
            "metrics"      : list of SaaS metric names queried
            "error"        : str (present only on failure)

    All other state fields are left untouched.
    """
    entity_name: str | None = state.get("entity_name") or None
    period_end = _now_utc()
    period_start = period_end - timedelta(days=_WINDOW_DAYS)

    try:
        with SessionLocal() as session:
            repo = KPIRepository(session)
            rows = repo.get_kpis_by_period(
                period_start=period_start,
                period_end=period_end,
                entity_name=entity_name,
            )
            records = [_serialize_row(r) for r in rows]

        saas_kpi_data: dict[str, Any] = {
            "records": records,
            "fetched_for": entity_name,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "metrics": sorted(_SAAS_METRICS),
        }

    except Exception as exc:  # noqa: BLE001
        saas_kpi_data = {
            "records": [],
            "fetched_for": entity_name,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "metrics": sorted(_SAAS_METRICS),
            "error": str(exc),
        }

    return {**state, "saas_kpi_data": saas_kpi_data}
