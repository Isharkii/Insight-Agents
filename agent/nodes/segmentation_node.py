"""
agent/nodes/segmentation_node.py

Segmentation Node: fetches the latest persisted segmentation snapshot
for the entity named in state.

No clustering, no ML, no feature engineering.
"""

from __future__ import annotations

from typing import Any

from agent.state import AgentState
from segmentation.repository import SegmentInsightRepository
from db.session import SessionLocal

_repository = SegmentInsightRepository()


def _serialize_row(row: Any) -> dict[str, Any]:
    """Convert a SegmentInsight ORM row to a plain JSON-safe dict."""
    return {
        "entity_name": row.entity_name,
        "period_end": row.period_end,
        "n_clusters": row.n_clusters,
        "segment_data": row.segment_data,
        "created_at": row.created_at.isoformat(),
    }


def segmentation_node(state: AgentState) -> AgentState:
    """
    LangGraph node: retrieve the latest segmentation snapshot for the entity.

    Reads:
        state["entity_name"] — entity whose segmentation is fetched.

    Writes:
        state["segmentation"] — dict with keys:
            "entity_name"  : str
            "period_end"   : str   ("YYYY-MM-DD")
            "n_clusters"   : int
            "segment_data" : dict  (labeled cluster profiles)
            "created_at"   : str   (ISO timestamp)
            "found"        : bool  (False when no record exists)
            "error"        : str   (present only on failure)
    """
    entity_name: str = state.get("entity_name") or ""

    try:
        with SessionLocal() as session:
            row = _repository.get_latest_segments(
                session=session,
                entity_name=entity_name,
            )

        if row is not None:
            segmentation: dict[str, Any] = {**_serialize_row(row), "found": True}
        else:
            segmentation = {
                "entity_name": entity_name,
                "period_end": None,
                "n_clusters": None,
                "segment_data": {},
                "created_at": None,
                "found": False,
            }

    except Exception as exc:  # noqa: BLE001
        segmentation = {
            "entity_name": entity_name,
            "period_end": None,
            "n_clusters": None,
            "segment_data": {},
            "created_at": None,
            "found": False,
            "error": str(exc),
        }

    return {**state, "segmentation": segmentation}
