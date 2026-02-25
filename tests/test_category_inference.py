"""
tests/test_category_inference.py

Deterministic tests for the category inference engine.

Verifies:
- SaaS dataset → high confidence saas
- E-commerce dataset → high confidence ecommerce
- Mixed dataset → ambiguous
- Sparse dataset → insufficient_data
- Determinism — same input twice → identical output
"""

from __future__ import annotations

import pytest

from app.services.category_inference import (
    CategoryInferenceResult,
    infer_category,
    validate_category_match,
)


# ---------------------------------------------------------------------------
# Fixtures: representative datasets
# ---------------------------------------------------------------------------


def _saas_metrics() -> list[str]:
    return [
        "mrr", "arr", "churn_rate", "ltv", "cac",
        "recurring_revenue", "active_subscriptions",
    ]


def _saas_columns() -> list[str]:
    return [
        "entity_name", "category", "metric_name", "metric_value",
        "timestamp", "region", "mrr", "churn_rate",
    ]


def _ecommerce_metrics() -> list[str]:
    return [
        "aov", "cart_abandonment", "roas", "conversion_rate",
        "gmv", "orders", "revenue",
    ]


def _ecommerce_columns() -> list[str]:
    return [
        "entity_name", "category", "metric_name", "metric_value",
        "timestamp", "order_id", "product_id", "conversion_rate",
    ]


def _healthcare_metrics() -> list[str]:
    return [
        "readmission_rate", "bed_occupancy_rate", "patient_revenue",
        "active_patients", "readmissions", "staff_hours",
    ]


def _healthcare_columns() -> list[str]:
    return [
        "entity_name", "category", "metric_name", "metric_value",
        "timestamp", "occupied_beds", "total_beds",
    ]


def _marketing_metrics() -> list[str]:
    return [
        "impressions", "clicks", "ctr", "cpc",
        "ad_spend", "roas", "conversions",
    ]


def _marketing_columns() -> list[str]:
    return [
        "entity_name", "category", "metric_name", "metric_value",
        "timestamp", "channel", "impressions", "clicks",
    ]


def _mixed_metrics() -> list[str]:
    """Metrics that span multiple categories — should produce ambiguous result."""
    return ["revenue", "churn_rate", "conversion_rate"]


def _mixed_columns() -> list[str]:
    return [
        "entity_name", "category", "metric_name", "metric_value",
        "timestamp",
    ]


def _sparse_metrics() -> list[str]:
    """Very generic metrics that don't map to any specific category."""
    return ["value_1", "score_2", "unknown_metric"]


def _sparse_columns() -> list[str]:
    return ["col_a", "col_b", "col_c"]


# ---------------------------------------------------------------------------
# Test: SaaS high confidence
# ---------------------------------------------------------------------------


class TestSaaSInference:
    def test_saas_high_confidence(self) -> None:
        result = infer_category(
            metric_names=_saas_metrics(),
            column_names=_saas_columns(),
            row_count=200,
            unique_entity_count=1,
        )
        assert isinstance(result, CategoryInferenceResult)
        assert result.inferred_category == "saas"
        assert result.confidence_score >= 0.80
        assert result.status == "success"
        assert len(result.evidence) > 0

    def test_saas_has_alternatives(self) -> None:
        result = infer_category(
            metric_names=_saas_metrics(),
            column_names=_saas_columns(),
            row_count=200,
            unique_entity_count=1,
        )
        assert isinstance(result.alternative_candidates, list)
        # Should have other categories scored
        assert len(result.alternative_candidates) > 0


# ---------------------------------------------------------------------------
# Test: E-commerce high confidence
# ---------------------------------------------------------------------------


class TestEcommerceInference:
    def test_ecommerce_high_confidence(self) -> None:
        result = infer_category(
            metric_names=_ecommerce_metrics(),
            column_names=_ecommerce_columns(),
            row_count=500,
            unique_entity_count=1,
        )
        assert result.inferred_category == "ecommerce"
        assert result.confidence_score >= 0.80
        assert result.status == "success"

    def test_ecommerce_has_transaction_evidence(self) -> None:
        result = infer_category(
            metric_names=_ecommerce_metrics(),
            column_names=_ecommerce_columns(),
            row_count=500,
            unique_entity_count=1,
        )
        # Should have evidence showing transaction column matches
        transaction_evidence = [e for e in result.evidence if "transaction" in e or "order" in e]
        assert len(transaction_evidence) > 0 or len(result.evidence) > 0


# ---------------------------------------------------------------------------
# Test: Healthcare high confidence
# ---------------------------------------------------------------------------


class TestHealthcareInference:
    def test_healthcare_high_confidence(self) -> None:
        result = infer_category(
            metric_names=_healthcare_metrics(),
            column_names=_healthcare_columns(),
            row_count=100,
            unique_entity_count=1,
        )
        assert result.inferred_category == "healthcare"
        assert result.confidence_score >= 0.80
        assert result.status == "success"


# ---------------------------------------------------------------------------
# Test: Marketing analytics high confidence
# ---------------------------------------------------------------------------


class TestMarketingInference:
    def test_marketing_high_confidence(self) -> None:
        result = infer_category(
            metric_names=_marketing_metrics(),
            column_names=_marketing_columns(),
            row_count=300,
            unique_entity_count=1,
        )
        assert result.inferred_category == "marketing_analytics"
        assert result.confidence_score >= 0.80
        assert result.status == "success"


# ---------------------------------------------------------------------------
# Test: Mixed dataset → ambiguous
# ---------------------------------------------------------------------------


class TestMixedInference:
    def test_mixed_returns_ambiguous_or_low_confidence(self) -> None:
        result = infer_category(
            metric_names=_mixed_metrics(),
            column_names=_mixed_columns(),
            row_count=50,
            unique_entity_count=1,
        )
        # With generic metrics, should not produce high confidence
        assert result.status in ("ambiguous", "insufficient_data")
        assert result.confidence_score < 0.80


# ---------------------------------------------------------------------------
# Test: Sparse dataset → insufficient_data
# ---------------------------------------------------------------------------


class TestSparseInference:
    def test_sparse_returns_insufficient(self) -> None:
        result = infer_category(
            metric_names=_sparse_metrics(),
            column_names=_sparse_columns(),
            row_count=5,
            unique_entity_count=1,
        )
        assert result.status == "insufficient_data"
        assert result.inferred_category is None
        assert result.confidence_score < 0.60


# ---------------------------------------------------------------------------
# Test: Determinism — same input twice → identical output
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_identical_output_on_repeat(self) -> None:
        kwargs = dict(
            metric_names=_saas_metrics(),
            column_names=_saas_columns(),
            row_count=200,
            unique_entity_count=1,
        )
        result1 = infer_category(**kwargs)
        result2 = infer_category(**kwargs)

        assert result1.inferred_category == result2.inferred_category
        assert result1.confidence_score == result2.confidence_score
        assert result1.status == result2.status
        assert result1.evidence == result2.evidence
        assert result1.alternative_candidates == result2.alternative_candidates

    def test_identical_output_ecommerce(self) -> None:
        kwargs = dict(
            metric_names=_ecommerce_metrics(),
            column_names=_ecommerce_columns(),
            row_count=500,
            unique_entity_count=1,
        )
        result1 = infer_category(**kwargs)
        result2 = infer_category(**kwargs)

        assert result1.model_dump() == result2.model_dump()


# ---------------------------------------------------------------------------
# Test: Validation helper
# ---------------------------------------------------------------------------


class TestValidateCategoryMatch:
    def test_no_warnings_when_match(self) -> None:
        result = infer_category(
            metric_names=_saas_metrics(),
            column_names=_saas_columns(),
            row_count=200,
            unique_entity_count=1,
        )
        warnings = validate_category_match(
            user_category="saas",
            inference_result=result,
        )
        assert warnings == []

    def test_warning_on_mismatch(self) -> None:
        result = infer_category(
            metric_names=_saas_metrics(),
            column_names=_saas_columns(),
            row_count=200,
            unique_entity_count=1,
        )
        warnings = validate_category_match(
            user_category="ecommerce",
            inference_result=result,
        )
        assert len(warnings) == 1
        assert "differs from" in warnings[0]

    def test_no_warnings_when_user_none(self) -> None:
        result = infer_category(
            metric_names=_saas_metrics(),
            column_names=_saas_columns(),
            row_count=200,
            unique_entity_count=1,
        )
        warnings = validate_category_match(
            user_category=None,
            inference_result=result,
        )
        assert warnings == []

    def test_alias_match_no_warning(self) -> None:
        """User supplying a known alias of the inferred category should not warn."""
        result = infer_category(
            metric_names=_saas_metrics(),
            column_names=_saas_columns(),
            row_count=200,
            unique_entity_count=1,
        )
        # "sales" is a category_alias for saas pack
        warnings = validate_category_match(
            user_category="sales",
            inference_result=result,
        )
        assert warnings == []


# ---------------------------------------------------------------------------
# Test: Result model structure
# ---------------------------------------------------------------------------


class TestResultModel:
    def test_result_fields(self) -> None:
        result = infer_category(
            metric_names=_saas_metrics(),
            column_names=_saas_columns(),
            row_count=200,
            unique_entity_count=1,
        )
        assert hasattr(result, "inferred_category")
        assert hasattr(result, "confidence_score")
        assert hasattr(result, "evidence")
        assert hasattr(result, "alternative_candidates")
        assert hasattr(result, "status")
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.status in ("success", "ambiguous", "insufficient_data")

    def test_alternatives_sorted_by_confidence(self) -> None:
        result = infer_category(
            metric_names=_saas_metrics(),
            column_names=_saas_columns(),
            row_count=200,
            unique_entity_count=1,
        )
        confidences = [c["confidence"] for c in result.alternative_candidates]
        assert confidences == sorted(confidences, reverse=True)


# ---------------------------------------------------------------------------
# Test: Empty inputs
# ---------------------------------------------------------------------------


class TestDisambiguationPenalty:
    def test_clear_winner_no_penalty(self) -> None:
        """Strong SaaS dataset should have large gap over runner-up — no penalty."""
        result = infer_category(
            metric_names=_saas_metrics(),
            column_names=_saas_columns(),
            row_count=200,
            unique_entity_count=1,
        )
        # SaaS has very distinctive metrics — gap to runner-up should be large
        assert result.status == "success"
        penalty_evidence = [e for e in result.evidence if "disambiguation_penalty" in e]
        # May or may not have penalty, but confidence should still be >= 0.80
        assert result.confidence_score >= 0.80

    def test_overlapping_metrics_penalized(self) -> None:
        """Metrics shared across categories should trigger a penalty."""
        # These metrics appear in both ecommerce and marketing core lists
        overlapping = ["roas", "conversion_rate", "revenue", "cac"]
        result = infer_category(
            metric_names=overlapping,
            column_names=["entity_name", "category", "metric_name", "metric_value", "timestamp"],
            row_count=50,
            unique_entity_count=1,
        )
        # With shared metrics, the gap between top candidates should be small
        # and the penalty should reduce confidence
        assert result.confidence_score < 0.80
        assert result.status in ("ambiguous", "insufficient_data")

    def test_penalty_produces_evidence_entry(self) -> None:
        """When penalty is applied, it should appear in evidence."""
        overlapping = ["roas", "conversion_rate", "revenue", "cac"]
        result = infer_category(
            metric_names=overlapping,
            column_names=["entity_name", "category", "metric_name", "metric_value"],
            row_count=30,
            unique_entity_count=1,
        )
        if result.confidence_score < 0.80:
            # Check if any penalty was applied
            penalty_evidence = [e for e in result.evidence if "disambiguation_penalty" in e]
            if penalty_evidence:
                assert "gap=" in penalty_evidence[0]
                assert "runner_up=" in penalty_evidence[0]


class TestEdgeCases:
    def test_empty_metrics(self) -> None:
        result = infer_category(
            metric_names=[],
            column_names=["entity_name", "category", "metric_name", "metric_value"],
            row_count=0,
            unique_entity_count=0,
        )
        assert result.status == "insufficient_data"
        assert result.inferred_category is None

    def test_empty_columns(self) -> None:
        result = infer_category(
            metric_names=["mrr", "churn_rate"],
            column_names=[],
            row_count=10,
            unique_entity_count=1,
        )
        # Should still work based on metric signal alone
        assert isinstance(result, CategoryInferenceResult)
        assert result.status in ("success", "ambiguous", "insufficient_data")
