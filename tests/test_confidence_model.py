from __future__ import annotations

from agent.helpers.confidence_model import (
    compute_standard_confidence,
    propagate_reasoning_strategy_confidence,
)


def test_compute_standard_confidence_applies_dataset_and_upstream_caps() -> None:
    result = compute_standard_confidence(
        values=[100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
        signals={"trend": 0.1},
        dataset_confidence=0.7,
        upstream_confidences=[0.9, 0.65],
    )

    assert result["confidence_score"] <= 0.65
    propagation = result["propagation"]
    assert propagation["dataset_cap"] == 0.7
    assert propagation["upstream_cap"] == 0.65


def test_compute_standard_confidence_penalizes_signal_conflicts() -> None:
    aligned = compute_standard_confidence(
        values=[100.0, 102.0, 105.0, 108.0, 111.0, 115.0],
        signals={"short_growth": 0.08, "long_growth": 0.05},
    )
    conflicted = compute_standard_confidence(
        values=[100.0, 102.0, 105.0, 108.0, 111.0, 115.0],
        signals={"short_growth": 0.08, "long_growth": -0.05},
    )

    assert conflicted["confidence_score"] < aligned["confidence_score"]


def test_reasoning_strategy_propagation_applies_strategy_penalty() -> None:
    propagated = propagate_reasoning_strategy_confidence(
        insight_confidence=0.82,
        reasoning_confidence=0.74,
        strategy_penalty=0.10,
    )
    assert propagated["insight_confidence"] == 0.82
    assert propagated["reasoning_confidence"] == 0.74
    assert propagated["strategy_confidence"] == 0.64
