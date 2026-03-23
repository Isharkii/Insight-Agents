"""
agent/nodes/segmentation_node.py

Segmentation Node: fetches the latest persisted segmentation snapshot
for the entity named in state.

No clustering, no ML, no feature engineering.

When no snapshot exists the node now returns a proper ``skipped`` envelope
so that signal_integrity and downstream consumers can count segmentation
as a missing analytical dimension rather than silently ignoring it.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.nodes.node_result import failed, skipped, success
from agent.state import AgentState
from app.failure_codes import OPTIONAL_FAILURES
from segmentation.repository import SegmentInsightRepository
from db.session import SessionLocal

_repository = SegmentInsightRepository()
logger = logging.getLogger(__name__)


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
        state["segmentation"] — envelope dict (success/skipped/failed).
    """
    entity_name: str = state.get("entity_name") or ""

    try:
        with SessionLocal() as session:
            row = _repository.get_latest_segments(
                session=session,
                entity_name=entity_name,
            )

        if row is not None:
            segmentation_payload: dict[str, Any] = {
                **_serialize_row(row),
                "found": True,
            }
            return {
                "segmentation": success(
                    segmentation_payload,
                    confidence_score=1.0,
                ),
            }

        # No snapshot found — propagate as a skipped envelope so
        # signal_integrity counts this as a missing dimension.
        if "missing_segmentation" in OPTIONAL_FAILURES:
            logger.warning(
                "Optional failure code=missing_segmentation entity=%r: "
                "no segmentation snapshot found",
                entity_name,
            )
        return {
            "segmentation": skipped(
                "no_segmentation_snapshot",
                {
                    "entity_name": entity_name,
                    "found": False,
                    "segment_data": {},
                },
            ),
        }

    except Exception as exc:  # noqa: BLE001
        if "missing_segmentation" in OPTIONAL_FAILURES:
            logger.warning(
                "Optional failure code=missing_segmentation entity=%r: %s",
                entity_name,
                exc,
            )
        return {
            "segmentation": failed(
                str(exc),
                {
                    "entity_name": entity_name,
                    "found": False,
                    "segment_data": {},
                },
            ),
        }
