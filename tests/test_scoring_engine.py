"""
tests/test_scoring_engine.py

Deterministic tests for the relative scoring engine.

Verifies:
- Percentile rank calculation
- Z-score computation and clipping
- Deviation % from median
- Sigmoid normalisation to 0–100
- Classification tiers
- Batch scoring
- Edge cases (empty benchmarks, flat distributions, outliers)
- Determinism — same input twice → identical output
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.services.scoring_engine import (
    CompositeScore,
    RelativeScore,
    classify,
    deviation_pct,
    normalise_to_100,
    percentile_rank,
    score_composite,
    score_metric,
    score_metric_macro_summary,
    score_metrics_batch,
    zscore,
)


# ---------------------------------------------------------------------------
# Benchmark fixture
# ---------------------------------------------------------------------------

_BENCHMARK = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]


# ---------------------------------------------------------------------------
# Test: percentile_rank
# ---------------------------------------------------------------------------


class TestPercentileRank:
    def test_below_all(self) -> None:
        bench = np.array(_BENCHMARK)
        assert percentile_rank(5.0, bench) == 0.0

    def test_above_all(self) -> None:
        bench = np.array(_BENCHMARK)
        assert percentile_rank(110.0, bench) == 100.0

    def test_at_median(self) -> None:
        bench = np.array(_BENCHMARK)
        pct = percentile_rank(55.0, bench)
        assert 40.0 <= pct <= 60.0

    def test_exact_match_value(self) -> None:
        bench = np.array(_BENCHMARK)
        # 50 is the 5th value; 4 values are strictly below → 40%
        assert percentile_rank(50.0, bench) == 40.0

    def test_empty_benchmark(self) -> None:
        assert percentile_rank(42.0, np.array([])) == 50.0


# ---------------------------------------------------------------------------
# Test: zscore
# ---------------------------------------------------------------------------


class TestZScore:
    def test_at_mean(self) -> None:
        assert zscore(55.0, 55.0, 10.0) == 0.0

    def test_one_std_above(self) -> None:
        z = zscore(65.0, 55.0, 10.0)
        assert abs(z - 1.0) < 1e-4

    def test_one_std_below(self) -> None:
        z = zscore(45.0, 55.0, 10.0)
        assert abs(z - (-1.0)) < 1e-4

    def test_clips_at_positive_bound(self) -> None:
        z = zscore(1000.0, 50.0, 10.0)
        assert z == 3.0

    def test_clips_at_negative_bound(self) -> None:
        z = zscore(-1000.0, 50.0, 10.0)
        assert z == -3.0

    def test_zero_std_uses_guard(self) -> None:
        # Should not raise; guard prevents division by zero
        z = zscore(55.0, 55.0, 0.0)
        assert math.isfinite(z)


# ---------------------------------------------------------------------------
# Test: deviation_pct
# ---------------------------------------------------------------------------


class TestDeviationPct:
    def test_at_median(self) -> None:
        assert deviation_pct(50.0, 50.0) == 0.0

    def test_above_median(self) -> None:
        dev = deviation_pct(75.0, 50.0)
        assert abs(dev - 50.0) < 0.01

    def test_below_median(self) -> None:
        dev = deviation_pct(25.0, 50.0)
        assert abs(dev - (-50.0)) < 0.01

    def test_zero_median_uses_guard(self) -> None:
        dev = deviation_pct(10.0, 0.0)
        assert math.isfinite(dev)
        assert dev > 0


# ---------------------------------------------------------------------------
# Test: normalise_to_100
# ---------------------------------------------------------------------------


class TestNormaliseTo100:
    def test_z_zero_gives_fifty(self) -> None:
        assert normalise_to_100(0.0) == 50.0

    def test_positive_z_above_fifty(self) -> None:
        assert normalise_to_100(1.0) > 50.0

    def test_negative_z_below_fifty(self) -> None:
        assert normalise_to_100(-1.0) < 50.0

    def test_extreme_positive(self) -> None:
        score = normalise_to_100(3.0)
        assert 90.0 <= score <= 100.0

    def test_extreme_negative(self) -> None:
        score = normalise_to_100(-3.0)
        assert 0.0 <= score <= 10.0

    def test_bounded_zero_to_hundred(self) -> None:
        for z in np.linspace(-5, 5, 100):
            s = normalise_to_100(float(z))
            assert 0.0 <= s <= 100.0


# ---------------------------------------------------------------------------
# Test: classify
# ---------------------------------------------------------------------------


class TestClassify:
    def test_top(self) -> None:
        assert classify(90.0) == "top"
        assert classify(80.0) == "top"

    def test_above_average(self) -> None:
        assert classify(70.0) == "above_average"
        assert classify(60.0) == "above_average"

    def test_average(self) -> None:
        assert classify(50.0) == "average"
        assert classify(40.0) == "average"

    def test_below_average(self) -> None:
        assert classify(30.0) == "below_average"
        assert classify(20.0) == "below_average"

    def test_bottom(self) -> None:
        assert classify(10.0) == "bottom"
        assert classify(0.0) == "bottom"


# ---------------------------------------------------------------------------
# Test: score_metric (integration of all primitives)
# ---------------------------------------------------------------------------


class TestScoreMetric:
    def test_returns_relative_score(self) -> None:
        result = score_metric("mrr", 55.0, _BENCHMARK)
        assert isinstance(result, RelativeScore)

    def test_high_performer(self) -> None:
        result = score_metric("mrr", 95.0, _BENCHMARK)
        assert result.percentile_rank >= 80.0
        assert result.z_score > 0
        assert result.deviation_pct > 0
        assert result.normalised_score >= 60.0
        assert result.classification in ("top", "above_average")

    def test_low_performer(self) -> None:
        result = score_metric("mrr", 15.0, _BENCHMARK)
        assert result.percentile_rank <= 20.0
        assert result.z_score < 0
        assert result.deviation_pct < 0
        assert result.normalised_score <= 40.0
        assert result.classification in ("below_average", "bottom")

    def test_average_performer(self) -> None:
        result = score_metric("mrr", 55.0, _BENCHMARK)
        assert 30.0 <= result.normalised_score <= 70.0
        assert result.classification in ("average", "above_average")

    def test_empty_benchmark(self) -> None:
        result = score_metric("mrr", 55.0, [])
        assert result.percentile_rank == 50.0
        assert result.normalised_score == 50.0
        assert result.classification == "average"

    def test_single_benchmark(self) -> None:
        result = score_metric("mrr", 100.0, [50.0])
        assert isinstance(result, RelativeScore)
        assert result.percentile_rank == 100.0
        # Single benchmark → zero std → neutral z-score/normalised score
        assert result.z_score == 0.0
        assert result.normalised_score == 50.0

    def test_benchmark_fields_populated(self) -> None:
        result = score_metric("churn_rate", 5.0, _BENCHMARK)
        assert result.metric_name == "churn_rate"
        assert result.client_value == 5.0
        assert result.benchmark_mean > 0
        assert result.benchmark_median > 0
        assert result.benchmark_std >= 0


# ---------------------------------------------------------------------------
# Test: score_metrics_batch
# ---------------------------------------------------------------------------


class TestScoreMetricsBatch:
    def test_batch_returns_dict(self) -> None:
        client = {"mrr": 80.0, "churn_rate": 5.0}
        benchmarks = {
            "mrr": _BENCHMARK,
            "churn_rate": [2.0, 3.0, 5.0, 8.0, 10.0],
        }
        results = score_metrics_batch(client, benchmarks)
        assert isinstance(results, dict)
        assert "mrr" in results
        assert "churn_rate" in results
        assert isinstance(results["mrr"], RelativeScore)

    def test_skips_missing_benchmark(self) -> None:
        client = {"mrr": 80.0, "unknown_metric": 10.0}
        benchmarks = {"mrr": _BENCHMARK}
        results = score_metrics_batch(client, benchmarks)
        assert "mrr" in results
        assert "unknown_metric" not in results

    def test_empty_inputs(self) -> None:
        assert score_metrics_batch({}, {}) == {}

    def test_skips_non_numeric_client(self) -> None:
        client = {"mrr": float("nan"), "arr": 100.0}
        benchmarks = {"mrr": _BENCHMARK, "arr": _BENCHMARK}
        results = score_metrics_batch(client, benchmarks)
        assert "mrr" not in results
        assert "arr" in results


# ---------------------------------------------------------------------------
# Test: Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_identical_output_on_repeat(self) -> None:
        r1 = score_metric("mrr", 75.0, _BENCHMARK)
        r2 = score_metric("mrr", 75.0, _BENCHMARK)
        assert r1.model_dump() == r2.model_dump()

    def test_batch_determinism(self) -> None:
        client = {"mrr": 80.0, "churn_rate": 5.0}
        benchmarks = {"mrr": _BENCHMARK, "churn_rate": [2.0, 3.0, 5.0, 8.0]}
        r1 = score_metrics_batch(client, benchmarks)
        r2 = score_metrics_batch(client, benchmarks)
        for key in r1:
            assert r1[key].model_dump() == r2[key].model_dump()


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_same_benchmark(self) -> None:
        """Flat distribution — std=0, guard prevents crash."""
        result = score_metric("mrr", 50.0, [50.0, 50.0, 50.0, 50.0])
        assert isinstance(result, RelativeScore)
        assert math.isfinite(result.z_score)
        assert result.normalised_score == 50.0

    def test_extreme_outlier_clipped(self) -> None:
        result = score_metric("mrr", 1_000_000.0, _BENCHMARK)
        assert result.z_score == 3.0
        assert result.normalised_score < 100.0

    def test_negative_values(self) -> None:
        bench = [-100.0, -50.0, 0.0, 50.0, 100.0]
        result = score_metric("growth", -75.0, bench)
        assert isinstance(result, RelativeScore)
        assert result.percentile_rank < 50.0

    def test_numpy_array_benchmark(self) -> None:
        bench = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = score_metric("mrr", 35.0, bench)
        assert isinstance(result, RelativeScore)


# ---------------------------------------------------------------------------
# Test: Composite scoring
# ---------------------------------------------------------------------------

# Shared benchmark data covering multiple categories
_COMPOSITE_BENCHMARKS: dict[str, list[float]] = {
    "growth_rate": [5.0, 10.0, 15.0, 20.0, 25.0],
    "churn_rate": [2.0, 5.0, 8.0, 12.0, 15.0],
    "mrr": [10_000.0, 20_000.0, 30_000.0, 40_000.0, 50_000.0],
    "conversion_rate": [1.0, 3.0, 5.0, 7.0, 10.0],
}


class TestCompositeScore:
    def test_returns_composite_model(self) -> None:
        client = {"growth_rate": 18.0, "churn_rate": 4.0, "mrr": 35_000.0, "conversion_rate": 6.0}
        result = score_composite(client, _COMPOSITE_BENCHMARKS)
        assert isinstance(result, CompositeScore)

    def test_all_categories_present(self) -> None:
        client = {"growth_rate": 18.0, "churn_rate": 4.0, "mrr": 35_000.0, "conversion_rate": 6.0}
        result = score_composite(client, _COMPOSITE_BENCHMARKS)
        assert "growth" in result.category_scores
        assert "retention" in result.category_scores
        assert "revenue" in result.category_scores
        assert "efficiency" in result.category_scores

    def test_overall_bounded_0_100(self) -> None:
        client = {"growth_rate": 18.0, "churn_rate": 4.0, "mrr": 35_000.0, "conversion_rate": 6.0}
        result = score_composite(client, _COMPOSITE_BENCHMARKS)
        assert 0.0 <= result.overall_score <= 100.0

    def test_category_scores_bounded(self) -> None:
        client = {"growth_rate": 18.0, "churn_rate": 4.0, "mrr": 35_000.0, "conversion_rate": 6.0}
        result = score_composite(client, _COMPOSITE_BENCHMARKS)
        for cat_score in result.category_scores.values():
            assert 0.0 <= cat_score <= 100.0

    def test_weakest_and_strongest_identified(self) -> None:
        client = {"growth_rate": 18.0, "churn_rate": 4.0, "mrr": 35_000.0, "conversion_rate": 6.0}
        result = score_composite(client, _COMPOSITE_BENCHMARKS)
        assert result.weakest_metric is not None
        assert result.strongest_metric is not None
        assert result.weakest_metric in result.metric_details
        assert result.strongest_metric in result.metric_details
        # Weakest should have lower score than strongest
        assert (
            result.metric_details[result.weakest_metric].normalised_score
            <= result.metric_details[result.strongest_metric].normalised_score
        )

    def test_custom_weights_shift_overall(self) -> None:
        client = {"growth_rate": 30.0, "churn_rate": 14.0, "mrr": 35_000.0, "conversion_rate": 6.0}
        # Default weights
        r_default = score_composite(client, _COMPOSITE_BENCHMARKS)
        # Heavily weight growth (where client is strong)
        r_growth = score_composite(
            client, _COMPOSITE_BENCHMARKS,
            category_weights={"growth": 0.90, "retention": 0.03, "revenue": 0.04, "efficiency": 0.03},
        )
        # Heavily weight retention (where client is weak — high churn)
        r_retention = score_composite(
            client, _COMPOSITE_BENCHMARKS,
            category_weights={"growth": 0.03, "retention": 0.90, "revenue": 0.04, "efficiency": 0.03},
        )
        # Growth-weighted score should be higher than retention-weighted
        assert r_growth.overall_score > r_retention.overall_score

    def test_custom_metric_categories(self) -> None:
        client = {"growth_rate": 20.0, "mrr": 40_000.0}
        custom_cats = {"my_group": ["growth_rate", "mrr"]}
        result = score_composite(
            client, _COMPOSITE_BENCHMARKS,
            metric_categories=custom_cats,
            category_weights={"my_group": 1.0},
        )
        assert "my_group" in result.category_scores
        assert "growth" not in result.category_scores

    def test_single_category(self) -> None:
        client = {"mrr": 35_000.0}
        result = score_composite(client, _COMPOSITE_BENCHMARKS)
        assert "revenue" in result.category_scores
        assert len(result.category_scores) == 1
        # Overall should equal that single category score
        assert result.overall_score == result.category_scores["revenue"]

    def test_empty_input_returns_neutral(self) -> None:
        result = score_composite({}, {})
        assert result.overall_score == 50.0
        assert result.category_scores == {}
        assert result.weakest_metric is None
        assert result.strongest_metric is None

    def test_metric_details_populated(self) -> None:
        client = {"growth_rate": 18.0, "mrr": 35_000.0}
        result = score_composite(client, _COMPOSITE_BENCHMARKS)
        assert "growth_rate" in result.metric_details
        assert "mrr" in result.metric_details
        assert isinstance(result.metric_details["growth_rate"], RelativeScore)

    def test_unrecognised_metrics_in_details_not_in_categories(self) -> None:
        """Metrics not in any category should appear in details but not affect category_scores."""
        client = {"growth_rate": 18.0, "exotic_metric": 42.0}
        benchmarks = {**_COMPOSITE_BENCHMARKS, "exotic_metric": [10.0, 20.0, 30.0, 40.0, 50.0]}
        result = score_composite(client, benchmarks)
        assert "exotic_metric" in result.metric_details
        # Only growth should appear as a category (exotic_metric has no category)
        assert "growth" in result.category_scores

    def test_determinism(self) -> None:
        client = {"growth_rate": 18.0, "churn_rate": 4.0, "mrr": 35_000.0}
        r1 = score_composite(client, _COMPOSITE_BENCHMARKS)
        r2 = score_composite(client, _COMPOSITE_BENCHMARKS)
        assert r1.model_dump() == r2.model_dump()


class TestMacroAdjustedScoring:
    def test_macro_adjusted_output_fields_present(self) -> None:
        result = score_metric(
            "revenue_growth",
            0.05,
            [0.03, 0.04, 0.05, 0.06],
            use_macro_adjustment=True,
            client_inflation_rate=0.10,
            benchmark_inflation_rate=[0.01, 0.01, 0.01, 0.01],
            adjustment_mode="growth",
        )
        payload = result.model_dump()
        assert "nominal_score" in payload
        assert "real_score" in payload
        assert "macro_resilience" in payload
        assert "delta_due_to_macro" in payload
        assert payload["macro_adjustment_applied"] is True

    def test_macro_adjustment_recomputes_ranking_using_real_values(self) -> None:
        nominal = score_metric(
            "revenue",
            105.0,
            [100.0, 102.0, 104.0],
            use_macro_adjustment=False,
        )
        adjusted = score_metric(
            "revenue",
            105.0,
            [100.0, 102.0, 104.0],
            use_macro_adjustment=True,
            client_inflation_rate=10.0,
            benchmark_inflation_rate=[1.0, 1.0, 1.0],
            adjustment_mode="level",
        )

        assert nominal.percentile_rank == 100.0
        assert adjusted.percentile_rank < nominal.percentile_rank
        assert adjusted.real_score < adjusted.nominal_score
        assert adjusted.delta_due_to_macro < 0

    def test_macro_toggle_off_keeps_nominal_scoring(self) -> None:
        result = score_metric(
            "revenue",
            105.0,
            [100.0, 102.0, 104.0],
            use_macro_adjustment=False,
            client_inflation_rate=10.0,
            benchmark_inflation_rate=[1.0, 1.0, 1.0],
            adjustment_mode="level",
        )
        assert result.nominal_score == result.real_score
        assert result.delta_due_to_macro == 0.0
        assert result.macro_resilience == 100.0
        assert result.macro_adjustment_applied is False

    def test_macro_adjustment_handles_missing_inflation_gracefully(self) -> None:
        result = score_metric(
            "revenue",
            105.0,
            [100.0, 102.0, 104.0],
            use_macro_adjustment=True,
        )
        assert result.macro_adjustment_applied is True
        assert result.nominal_score == result.real_score
        assert result.delta_due_to_macro == 0.0

    def test_macro_summary_output_shape(self) -> None:
        summary = score_metric_macro_summary(
            "revenue_growth",
            0.05,
            [0.03, 0.04, 0.05, 0.06],
            use_macro_adjustment=True,
            client_inflation_rate=0.10,
            benchmark_inflation_rate=[0.01, 0.01, 0.01, 0.01],
            adjustment_mode="growth",
        )
        assert set(summary.keys()) == {
            "nominal_score",
            "real_score",
            "macro_resilience",
            "delta_due_to_macro",
        }
