"""
tests/test_classification.py

Deterministic tests for the classification logic module.

Verifies:
- Default tier classification at each band
- Boundary behaviour
- Confidence calculation (centre vs edge of band)
- Relative position descriptors
- Custom tier overrides
- Clamping of out-of-range scores
- Determinism
"""

from __future__ import annotations

import pytest

from app.services.classification import (
    ClassificationResult,
    classify_score,
)


# ---------------------------------------------------------------------------
# Test: Default tier labels
# ---------------------------------------------------------------------------


class TestDefaultTiers:
    def test_market_leader(self) -> None:
        result = classify_score(92.0)
        assert result.classification == "Market Leader"

    def test_market_leader_lower_bound(self) -> None:
        result = classify_score(85.0)
        assert result.classification == "Market Leader"

    def test_strong_performer(self) -> None:
        result = classify_score(77.0)
        assert result.classification == "Strong Performer"

    def test_strong_performer_lower_bound(self) -> None:
        result = classify_score(70.0)
        assert result.classification == "Strong Performer"

    def test_competitive(self) -> None:
        result = classify_score(60.0)
        assert result.classification == "Competitive"

    def test_competitive_lower_bound(self) -> None:
        result = classify_score(50.0)
        assert result.classification == "Competitive"

    def test_underperforming(self) -> None:
        result = classify_score(40.0)
        assert result.classification == "Underperforming"

    def test_underperforming_lower_bound(self) -> None:
        result = classify_score(30.0)
        assert result.classification == "Underperforming"

    def test_critical_risk(self) -> None:
        result = classify_score(15.0)
        assert result.classification == "Critical Risk"

    def test_critical_risk_zero(self) -> None:
        result = classify_score(0.0)
        assert result.classification == "Critical Risk"

    def test_perfect_score(self) -> None:
        result = classify_score(100.0)
        assert result.classification == "Market Leader"


# ---------------------------------------------------------------------------
# Test: Confidence calculation
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_centre_of_band_high_confidence(self) -> None:
        # Market Leader band: 85–100, centre = 92.5
        result = classify_score(92.5)
        assert result.confidence == 1.0

    def test_edge_of_band_low_confidence(self) -> None:
        # Score right at the lower bound of Strong Performer (70)
        result = classify_score(70.0)
        assert result.confidence == 0.0

    def test_near_boundary_low_confidence(self) -> None:
        # Just above boundary (70.5 in 70–85 band)
        result = classify_score(70.5)
        assert result.confidence < 0.2

    def test_deep_inside_band(self) -> None:
        # 60 in Competitive band (50–70), centre = 60
        result = classify_score(60.0)
        assert result.confidence == 1.0

    def test_confidence_bounded_0_1(self) -> None:
        for score in [0.0, 15.0, 30.0, 50.0, 70.0, 85.0, 100.0]:
            result = classify_score(score)
            assert 0.0 <= result.confidence <= 1.0

    def test_score_at_100_confidence(self) -> None:
        # 100 is at the upper edge of Market Leader (85–100)
        result = classify_score(100.0)
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Test: Relative position
# ---------------------------------------------------------------------------


class TestRelativePosition:
    def test_top_10(self) -> None:
        result = classify_score(95.0)
        assert result.relative_position == "Top 10%"

    def test_top_25(self) -> None:
        result = classify_score(80.0)
        assert result.relative_position == "Top 25%"

    def test_top_50(self) -> None:
        result = classify_score(60.0)
        assert result.relative_position == "Top 50%"

    def test_bottom_50(self) -> None:
        result = classify_score(30.0)
        assert result.relative_position == "Bottom 50%"

    def test_bottom_25(self) -> None:
        result = classify_score(10.0)
        assert result.relative_position == "Bottom 25%"


# ---------------------------------------------------------------------------
# Test: Custom tiers
# ---------------------------------------------------------------------------


class TestCustomTiers:
    def test_three_tier_system(self) -> None:
        custom = [(70.0, "High"), (40.0, "Medium"), (0.0, "Low")]
        assert classify_score(80.0, tiers=custom).classification == "High"
        assert classify_score(55.0, tiers=custom).classification == "Medium"
        assert classify_score(20.0, tiers=custom).classification == "Low"

    def test_two_tier_system(self) -> None:
        custom = [(50.0, "Pass"), (0.0, "Fail")]
        assert classify_score(75.0, tiers=custom).classification == "Pass"
        assert classify_score(25.0, tiers=custom).classification == "Fail"

    def test_custom_tiers_confidence(self) -> None:
        custom = [(50.0, "Pass"), (0.0, "Fail")]
        # Centre of Pass band (50–100) = 75
        result = classify_score(75.0, tiers=custom)
        assert result.confidence == 1.0


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_clamp_above_100(self) -> None:
        result = classify_score(150.0)
        assert result.score == 100.0
        assert result.classification == "Market Leader"

    def test_clamp_below_0(self) -> None:
        result = classify_score(-10.0)
        assert result.score == 0.0
        assert result.classification == "Critical Risk"

    def test_returns_classification_result(self) -> None:
        result = classify_score(50.0)
        assert isinstance(result, ClassificationResult)

    def test_result_is_frozen(self) -> None:
        result = classify_score(50.0)
        with pytest.raises(Exception):
            result.classification = "Hacked"


# ---------------------------------------------------------------------------
# Test: Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_identical_output_on_repeat(self) -> None:
        r1 = classify_score(72.4)
        r2 = classify_score(72.4)
        assert r1.model_dump() == r2.model_dump()

    def test_determinism_with_custom_tiers(self) -> None:
        custom = [(60.0, "Good"), (0.0, "Bad")]
        r1 = classify_score(45.0, tiers=custom)
        r2 = classify_score(45.0, tiers=custom)
        assert r1.model_dump() == r2.model_dump()
