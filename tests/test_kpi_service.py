"""
tests/test_kpi_service.py

Pytest unit tests for KPIService.

All tests are pure Python — no database, no I/O, mock inputs only.
Every assertion is deterministic: given the same inputs, the same
output must be produced every time.

Coverage
--------
- Normal MRR calculation
- Zero revenue edge case
- Zero churn edge case
- High churn scenario
- Growth rate negative scenario
- Growth rate division by zero handling
- LTV normal and division-by-zero
- KPIResult structure contracts
- Statelessness across multiple calls
"""

from __future__ import annotations

import pytest

from app.services.kpi_service import (
    ChurnInput,
    GrowthRateInput,
    KPIResult,
    KPIService,
    LTVInput,
    MRRInput,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def svc() -> KPIService:
    """Fresh KPIService instance for each test."""
    return KPIService()


# ---------------------------------------------------------------------------
# KPIResult contract
# ---------------------------------------------------------------------------


class TestKPIResultContract:
    def test_is_frozen(self) -> None:
        result = KPIResult(metric="mrr", value=1000.0, unit="currency")
        with pytest.raises((AttributeError, TypeError)):
            result.value = 999.0  # type: ignore[misc]

    def test_error_defaults_to_none(self) -> None:
        result = KPIResult(metric="mrr", value=100.0, unit="currency")
        assert result.error is None

    def test_computed_at_is_populated(self) -> None:
        result = KPIResult(metric="mrr", value=0.0, unit="currency")
        assert result.computed_at is not None

    def test_value_can_be_none_for_failed_calculations(self) -> None:
        result = KPIResult(metric="growth_rate", value=None, unit="rate", error="div/0")
        assert result.value is None
        assert result.error is not None


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------


class TestMRR:
    """calculate_mrr — normal MRR calculation and zero revenue edge cases."""

    # --- Normal MRR calculation ---

    def test_normal_mrr_sums_revenues(self, svc: KPIService) -> None:
        """Standard case: multiple active subscriptions are summed."""
        result = svc.calculate_mrr(MRRInput(active_subscription_revenues=[500.0, 300.0, 200.0]))
        assert result.value == pytest.approx(1000.0)

    def test_normal_mrr_single_subscription(self, svc: KPIService) -> None:
        result = svc.calculate_mrr(MRRInput(active_subscription_revenues=[750.0]))
        assert result.value == pytest.approx(750.0)

    def test_normal_mrr_fractional_values(self, svc: KPIService) -> None:
        result = svc.calculate_mrr(MRRInput(active_subscription_revenues=[99.99, 0.01]))
        assert result.value == pytest.approx(100.0)

    @pytest.mark.parametrize(
        "revenues, expected",
        [
            ([100.0, 200.0, 300.0], 600.0),
            ([1_000.0, 2_500.0], 3_500.0),
            ([49.99, 49.99, 49.99], pytest.approx(149.97, rel=1e-6)),
        ],
    )
    def test_normal_mrr_parametrized(
        self, svc: KPIService, revenues: list[float], expected: float
    ) -> None:
        assert svc.calculate_mrr(MRRInput(active_subscription_revenues=revenues)).value == expected

    # --- Zero revenue edge case ---

    def test_zero_revenue_empty_list_returns_zero(self, svc: KPIService) -> None:
        """No active subscriptions — valid state, MRR is 0.0, not an error."""
        result = svc.calculate_mrr(MRRInput(active_subscription_revenues=[]))
        assert result.value == 0.0
        assert result.error is None

    def test_zero_revenue_single_zero_subscription(self, svc: KPIService) -> None:
        """One subscription with £0 revenue — still not an error."""
        result = svc.calculate_mrr(MRRInput(active_subscription_revenues=[0.0]))
        assert result.value == 0.0
        assert result.error is None

    def test_zero_revenue_all_subscriptions_zero(self, svc: KPIService) -> None:
        result = svc.calculate_mrr(MRRInput(active_subscription_revenues=[0.0, 0.0, 0.0]))
        assert result.value == 0.0

    # --- Result metadata ---

    def test_metric_name_is_mrr(self, svc: KPIService) -> None:
        assert svc.calculate_mrr(MRRInput(active_subscription_revenues=[])).metric == "mrr"

    def test_unit_is_currency(self, svc: KPIService) -> None:
        assert svc.calculate_mrr(MRRInput(active_subscription_revenues=[1.0])).unit == "currency"


# ---------------------------------------------------------------------------
# Churn
# ---------------------------------------------------------------------------


class TestChurn:
    """calculate_churn — zero churn edge case and high churn scenario."""

    # --- Zero churn edge case ---

    def test_zero_churn_no_customers_lost(self, svc: KPIService) -> None:
        """All customers retained — churn is 0.0, not an error."""
        result = svc.calculate_churn(ChurnInput(customers_at_start=200, customers_lost=0))
        assert result.value == pytest.approx(0.0)
        assert result.error is None

    def test_zero_churn_is_not_a_failure(self, svc: KPIService) -> None:
        result = svc.calculate_churn(ChurnInput(customers_at_start=1, customers_lost=0))
        assert result.value is not None

    # --- Normal churn ---

    def test_standard_churn_five_percent(self, svc: KPIService) -> None:
        """10 lost from 200 → 5 % churn rate."""
        result = svc.calculate_churn(ChurnInput(customers_at_start=200, customers_lost=10))
        assert result.value == pytest.approx(0.05)

    def test_churn_one_from_one(self, svc: KPIService) -> None:
        result = svc.calculate_churn(ChurnInput(customers_at_start=1, customers_lost=1))
        assert result.value == pytest.approx(1.0)

    # --- High churn scenario ---

    def test_high_churn_half_lost(self, svc: KPIService) -> None:
        """50 % churn — still a valid, computable result."""
        result = svc.calculate_churn(ChurnInput(customers_at_start=100, customers_lost=50))
        assert result.value == pytest.approx(0.5)
        assert result.error is None

    def test_high_churn_full_loss(self, svc: KPIService) -> None:
        """100 % churn — entire customer base churned."""
        result = svc.calculate_churn(ChurnInput(customers_at_start=50, customers_lost=50))
        assert result.value == pytest.approx(1.0)
        assert result.error is None

    def test_high_churn_near_total(self, svc: KPIService) -> None:
        result = svc.calculate_churn(ChurnInput(customers_at_start=1000, customers_lost=980))
        assert result.value == pytest.approx(0.98)

    # --- Division-by-zero guard ---

    def test_zero_customers_at_start_returns_none_value(self, svc: KPIService) -> None:
        result = svc.calculate_churn(ChurnInput(customers_at_start=0, customers_lost=0))
        assert result.value is None

    def test_zero_customers_at_start_sets_error(self, svc: KPIService) -> None:
        result = svc.calculate_churn(ChurnInput(customers_at_start=0, customers_lost=0))
        assert result.error is not None
        assert "zero" in result.error.lower()

    def test_zero_customers_at_start_with_nonzero_lost(self, svc: KPIService) -> None:
        """Incoherent input still handled without raising."""
        result = svc.calculate_churn(ChurnInput(customers_at_start=0, customers_lost=5))
        assert result.value is None
        assert result.error is not None

    # --- Parametrized rates ---

    @pytest.mark.parametrize(
        "at_start, lost, expected_rate",
        [
            (100,  5,  0.05),
            (200, 10,  0.05),
            (50,  25,  0.5),
            (1000, 1,  0.001),
        ],
    )
    def test_churn_rate_formula(
        self,
        svc: KPIService,
        at_start: int,
        lost: int,
        expected_rate: float,
    ) -> None:
        result = svc.calculate_churn(ChurnInput(customers_at_start=at_start, customers_lost=lost))
        assert result.value == pytest.approx(expected_rate, rel=1e-6)

    # --- Result metadata ---

    def test_metric_name_is_churn_rate(self, svc: KPIService) -> None:
        result = svc.calculate_churn(ChurnInput(customers_at_start=10, customers_lost=1))
        assert result.metric == "churn_rate"

    def test_unit_is_rate(self, svc: KPIService) -> None:
        result = svc.calculate_churn(ChurnInput(customers_at_start=10, customers_lost=1))
        assert result.unit == "rate"


# ---------------------------------------------------------------------------
# LTV
# ---------------------------------------------------------------------------


class TestLTV:
    """calculate_ltv — standard LTV and division-by-zero on zero churn."""

    def test_standard_ltv(self, svc: KPIService) -> None:
        """ARPU=100, churn=0.05 → LTV=2000."""
        result = svc.calculate_ltv(LTVInput(average_revenue_per_user=100.0, churn_rate=0.05))
        assert result.value == pytest.approx(2000.0)

    def test_ltv_equals_arpu_when_full_churn(self, svc: KPIService) -> None:
        """churn_rate=1.0 → LTV = ARPU (customer retained exactly one period)."""
        result = svc.calculate_ltv(LTVInput(average_revenue_per_user=200.0, churn_rate=1.0))
        assert result.value == pytest.approx(200.0)

    def test_high_churn_produces_low_ltv(self, svc: KPIService) -> None:
        """50 % churn — LTV is only 2× ARPU."""
        result = svc.calculate_ltv(LTVInput(average_revenue_per_user=50.0, churn_rate=0.5))
        assert result.value == pytest.approx(100.0)

    def test_low_churn_produces_high_ltv(self, svc: KPIService) -> None:
        """1 % churn — LTV is 100× ARPU."""
        result = svc.calculate_ltv(LTVInput(average_revenue_per_user=100.0, churn_rate=0.01))
        assert result.value == pytest.approx(10_000.0)

    # --- Division-by-zero on zero churn ---

    def test_zero_churn_rate_value_is_none(self, svc: KPIService) -> None:
        """churn_rate=0 implies infinite LTV — undefined, not a number."""
        result = svc.calculate_ltv(LTVInput(average_revenue_per_user=100.0, churn_rate=0.0))
        assert result.value is None

    def test_zero_churn_rate_sets_error(self, svc: KPIService) -> None:
        result = svc.calculate_ltv(LTVInput(average_revenue_per_user=100.0, churn_rate=0.0))
        assert result.error is not None
        assert "zero" in result.error.lower()

    def test_zero_churn_rate_does_not_raise(self, svc: KPIService) -> None:
        """Must return a result, never raise ZeroDivisionError."""
        result = svc.calculate_ltv(LTVInput(average_revenue_per_user=500.0, churn_rate=0.0))
        assert isinstance(result, KPIResult)

    # --- Result metadata ---

    def test_metric_name_is_ltv(self, svc: KPIService) -> None:
        result = svc.calculate_ltv(LTVInput(average_revenue_per_user=50.0, churn_rate=0.1))
        assert result.metric == "ltv"

    def test_unit_is_currency(self, svc: KPIService) -> None:
        result = svc.calculate_ltv(LTVInput(average_revenue_per_user=50.0, churn_rate=0.1))
        assert result.unit == "currency"


# ---------------------------------------------------------------------------
# Growth Rate
# ---------------------------------------------------------------------------


class TestGrowthRate:
    """calculate_growth_rate — negative scenarios and division-by-zero handling."""

    # --- Normal positive growth ---

    def test_positive_growth_25_percent(self, svc: KPIService) -> None:
        """1000 → 1250 = +25 %."""
        result = svc.calculate_growth_rate(
            GrowthRateInput(current_period_revenue=1250.0, previous_period_revenue=1000.0)
        )
        assert result.value == pytest.approx(0.25)

    def test_100_percent_growth(self, svc: KPIService) -> None:
        result = svc.calculate_growth_rate(
            GrowthRateInput(current_period_revenue=2000.0, previous_period_revenue=1000.0)
        )
        assert result.value == pytest.approx(1.0)

    def test_zero_growth_same_revenue(self, svc: KPIService) -> None:
        result = svc.calculate_growth_rate(
            GrowthRateInput(current_period_revenue=500.0, previous_period_revenue=500.0)
        )
        assert result.value == pytest.approx(0.0)
        assert result.error is None

    # --- Growth rate negative scenario ---

    def test_negative_growth_revenue_declined(self, svc: KPIService) -> None:
        """1000 → 800 = -20 % — declining revenue is a valid, negative rate."""
        result = svc.calculate_growth_rate(
            GrowthRateInput(current_period_revenue=800.0, previous_period_revenue=1000.0)
        )
        assert result.value == pytest.approx(-0.20)

    def test_negative_growth_severe_decline(self, svc: KPIService) -> None:
        """Revenue dropped by 75 %."""
        result = svc.calculate_growth_rate(
            GrowthRateInput(current_period_revenue=250.0, previous_period_revenue=1000.0)
        )
        assert result.value == pytest.approx(-0.75)

    def test_negative_growth_near_zero_current(self, svc: KPIService) -> None:
        """Almost all revenue lost."""
        result = svc.calculate_growth_rate(
            GrowthRateInput(current_period_revenue=1.0, previous_period_revenue=1000.0)
        )
        assert result.value == pytest.approx(-0.999, rel=1e-3)
        assert result.error is None

    def test_negative_growth_result_has_no_error(self, svc: KPIService) -> None:
        """Negative growth is a computable outcome — error must be None."""
        result = svc.calculate_growth_rate(
            GrowthRateInput(current_period_revenue=100.0, previous_period_revenue=200.0)
        )
        assert result.error is None

    @pytest.mark.parametrize(
        "current, previous, expected",
        [
            (800.0,  1000.0, -0.20),
            (500.0,  1000.0, -0.50),
            (100.0,  1000.0, -0.90),
            (1100.0, 1000.0,  0.10),
        ],
    )
    def test_growth_rate_formula(
        self,
        svc: KPIService,
        current: float,
        previous: float,
        expected: float,
    ) -> None:
        result = svc.calculate_growth_rate(
            GrowthRateInput(current_period_revenue=current, previous_period_revenue=previous)
        )
        assert result.value == pytest.approx(expected, rel=1e-6)

    # --- Growth rate division by zero handling ---

    def test_zero_previous_revenue_value_is_none(self, svc: KPIService) -> None:
        """Previous period is zero — growth rate is mathematically undefined."""
        result = svc.calculate_growth_rate(
            GrowthRateInput(current_period_revenue=500.0, previous_period_revenue=0.0)
        )
        assert result.value is None

    def test_zero_previous_revenue_sets_error(self, svc: KPIService) -> None:
        result = svc.calculate_growth_rate(
            GrowthRateInput(current_period_revenue=500.0, previous_period_revenue=0.0)
        )
        assert result.error is not None
        assert "zero" in result.error.lower()

    def test_zero_previous_revenue_does_not_raise(self, svc: KPIService) -> None:
        """Must return a KPIResult, never raise ZeroDivisionError."""
        result = svc.calculate_growth_rate(
            GrowthRateInput(current_period_revenue=500.0, previous_period_revenue=0.0)
        )
        assert isinstance(result, KPIResult)

    def test_both_periods_zero_value_is_none(self, svc: KPIService) -> None:
        result = svc.calculate_growth_rate(
            GrowthRateInput(current_period_revenue=0.0, previous_period_revenue=0.0)
        )
        assert result.value is None

    def test_zero_current_nonzero_previous_is_computable(self, svc: KPIService) -> None:
        """Current revenue of zero is valid — means 100 % decline."""
        result = svc.calculate_growth_rate(
            GrowthRateInput(current_period_revenue=0.0, previous_period_revenue=1000.0)
        )
        assert result.value == pytest.approx(-1.0)
        assert result.error is None

    # --- Result metadata ---

    def test_metric_name_is_growth_rate(self, svc: KPIService) -> None:
        result = svc.calculate_growth_rate(
            GrowthRateInput(current_period_revenue=110.0, previous_period_revenue=100.0)
        )
        assert result.metric == "growth_rate"

    def test_unit_is_rate(self, svc: KPIService) -> None:
        result = svc.calculate_growth_rate(
            GrowthRateInput(current_period_revenue=110.0, previous_period_revenue=100.0)
        )
        assert result.unit == "rate"


# ---------------------------------------------------------------------------
# Statelessness
# ---------------------------------------------------------------------------


class TestStatelessness:
    """KPIService must produce identical results regardless of call order or instance."""

    def test_repeated_mrr_calls_are_independent(self, svc: KPIService) -> None:
        r1 = svc.calculate_mrr(MRRInput(active_subscription_revenues=[100.0]))
        r2 = svc.calculate_mrr(MRRInput(active_subscription_revenues=[200.0]))
        assert r1.value == pytest.approx(100.0)
        assert r2.value == pytest.approx(200.0)

    def test_separate_instances_produce_identical_results(self) -> None:
        data = ChurnInput(customers_at_start=100, customers_lost=5)
        assert KPIService().calculate_churn(data).value == KPIService().calculate_churn(data).value

    def test_mrr_then_growth_rate_do_not_interfere(self, svc: KPIService) -> None:
        mrr = svc.calculate_mrr(MRRInput(active_subscription_revenues=[1000.0]))
        growth = svc.calculate_growth_rate(
            GrowthRateInput(current_period_revenue=1100.0, previous_period_revenue=1000.0)
        )
        assert mrr.value == pytest.approx(1000.0)
        assert growth.value == pytest.approx(0.10)
