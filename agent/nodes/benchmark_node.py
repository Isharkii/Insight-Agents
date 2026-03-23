"""
agent/nodes/benchmark_node.py

Competitive Benchmark Node: runs the deterministic peer benchmarking
pipeline within the agent graph to produce direction-aware rankings,
composite scoring, and market positioning classification.

This wires the existing ``competitive_benchmark_service`` into the
LangGraph pipeline so benchmark results feed into risk scoring,
signal integrity, and LLM synthesis.

All computations are deterministic — no LLM, no external APIs.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.helpers.confidence_model import compute_standard_confidence
from agent.nodes.node_result import failed, skipped, success
from agent.state import AgentState
from db.session import SessionLocal

logger = logging.getLogger(__name__)


# ── Market Positioning Classification ────────────────────────────────────
#
# Uses composite score components to classify entity position:
#   Leader     — high overall + positive growth trajectory
#   Challenger — moderate overall but strong growth (catching up)
#   Stable     — moderate overall + flat/moderate growth
#   Declining  — low overall OR negative growth trajectory
#
# Thresholds are intentionally conservative.

_LEADER_OVERALL_MIN = 65.0
_LEADER_GROWTH_MIN = 55.0

_CHALLENGER_GROWTH_MIN = 60.0
_CHALLENGER_OVERALL_MIN = 40.0

_DECLINING_OVERALL_MAX = 35.0
_DECLINING_GROWTH_MAX = 40.0


def classify_market_position(
    overall_score: float,
    growth_score: float,
    stability_score: float,
) -> dict[str, Any]:
    """Classify market position from composite score components.

    Returns a dict with:
        position: str — one of Leader, Challenger, Stable, Declining
        confidence: float — classification confidence (0-1)
        drivers: dict — score components that drove the classification
    """
    drivers = {
        "overall_score": round(overall_score, 2),
        "growth_score": round(growth_score, 2),
        "stability_score": round(stability_score, 2),
    }

    # Leader: strong across the board with positive growth
    if overall_score >= _LEADER_OVERALL_MIN and growth_score >= _LEADER_GROWTH_MIN:
        margin = min(
            (overall_score - _LEADER_OVERALL_MIN) / 35.0,
            (growth_score - _LEADER_GROWTH_MIN) / 45.0,
        )
        return {
            "position": "Leader",
            "confidence": round(min(1.0, 0.7 + margin * 0.3), 4),
            "drivers": drivers,
        }

    # Declining: weak overall or growth collapsing
    if overall_score <= _DECLINING_OVERALL_MAX or growth_score <= _DECLINING_GROWTH_MAX:
        severity = max(
            (_DECLINING_OVERALL_MAX - overall_score) / _DECLINING_OVERALL_MAX
            if overall_score <= _DECLINING_OVERALL_MAX
            else 0.0,
            (_DECLINING_GROWTH_MAX - growth_score) / _DECLINING_GROWTH_MAX
            if growth_score <= _DECLINING_GROWTH_MAX
            else 0.0,
        )
        return {
            "position": "Declining",
            "confidence": round(min(1.0, 0.6 + severity * 0.4), 4),
            "drivers": drivers,
        }

    # Challenger: moderate overall but strong growth momentum
    if growth_score >= _CHALLENGER_GROWTH_MIN and overall_score >= _CHALLENGER_OVERALL_MIN:
        growth_edge = (growth_score - overall_score) / 100.0
        return {
            "position": "Challenger",
            "confidence": round(min(1.0, 0.6 + max(0.0, growth_edge)), 4),
            "drivers": drivers,
        }

    # Stable: everything in the middle
    return {
        "position": "Stable",
        "confidence": round(
            min(1.0, 0.5 + stability_score / 200.0),
            4,
        ),
        "drivers": drivers,
    }


def benchmark_node(state: AgentState) -> AgentState:
    """LangGraph node: run deterministic peer benchmarking.

    Reads:
        state["entity_name"]   — entity to benchmark
        state["business_type"] — selects category for peer sourcing

    Writes:
        state["benchmark_data"] — envelope with ranking, composite, positioning
    """
    entity_name = str(state.get("entity_name") or "").strip()
    business_type = str(state.get("business_type") or "").strip().lower()

    if not entity_name:
        return {
            "benchmark_data": skipped(
                "missing_entity_name",
                {"entity_name": ""},
            ),
        }

    try:
        from app.services.competitive_benchmark_service import (
            build_competitive_benchmark_snapshot,
        )

        with SessionLocal() as session:
            snapshot = build_competitive_benchmark_snapshot(
                db=session,
                entity_name=entity_name,
                business_type=business_type,
            )

        status = str(snapshot.get("status") or "").strip().lower()

        if status == "skipped":
            return {
                "benchmark_data": skipped(
                    snapshot.get("reason", "benchmark_skipped"),
                    snapshot,
                ),
            }

        if status == "partial":
            # Partial = some data but not enough for full benchmark.
            # Still return what we have so signal_integrity can count it.
            return {
                "benchmark_data": success(
                    snapshot,
                    warnings=[
                        f"Benchmark partial: {snapshot.get('reason', 'insufficient_peer_data')}",
                    ],
                    confidence_score=0.2,
                ),
            }

        # Full benchmark — extract composite and classify position
        composite = snapshot.get("composite") or {}
        ranking = snapshot.get("ranking") or {}

        overall_score = float(composite.get("overall_score", 50.0))
        growth_score = float(composite.get("growth_score", 50.0))
        stability_score = float(composite.get("stability_score", 50.0))

        market_position = classify_market_position(
            overall_score=overall_score,
            growth_score=growth_score,
            stability_score=stability_score,
        )
        snapshot["market_position"] = market_position

        # Peer count for confidence calculation
        peer_selection = snapshot.get("peer_selection") or {}
        selected_peers = peer_selection.get("selected_peers") or []
        peer_count = len(selected_peers)

        # Compute confidence from benchmark quality signals
        confidence_model = compute_standard_confidence(
            values=[overall_score, growth_score, stability_score],
            signals={
                "peer_count": min(1.0, peer_count / 5.0),
                "overall_score_spread": -(abs(overall_score - 50.0) / 50.0),
                "position_confidence": market_position.get("confidence", 0.5),
            },
            dataset_confidence=1.0,
            upstream_confidences=[],
            status="success",
        )

        benchmark_confidence = float(confidence_model["confidence_score"])

        logger.info(
            "Benchmark for %s: position=%s overall=%.1f growth=%.1f "
            "peers=%d confidence=%.2f",
            entity_name,
            market_position["position"],
            overall_score,
            growth_score,
            peer_count,
            benchmark_confidence,
        )

        return {
            "benchmark_data": success(
                snapshot,
                warnings=confidence_model.get("warnings", []),
                confidence_score=benchmark_confidence,
            ),
        }

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Benchmark node failed for %s: %s",
            entity_name,
            exc,
            exc_info=True,
        )
        return {
            "benchmark_data": failed(
                str(exc),
                {"entity_name": entity_name},
            ),
        }
