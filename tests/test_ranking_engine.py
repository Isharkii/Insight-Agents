"""
tests/test_ranking_engine.py

Deterministic tests for the competitive ranking engine.

Verifies:
- Standard competition ranking (1224 pattern)
- Tie handling
- Per-metric ranking
- Overall competitive ranking
- Tier classification
- Edge cases (single participant, missing metrics)
- Determinism
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.ranking_engine import (
    CompetitiveRanking,
    MetricRank,
    competition_rank,
    rank_competitive,
    rank_metric,
)


# ---------------------------------------------------------------------------
# Test: competition_rank primitive
# ---------------------------------------------------------------------------


class TestCompetitionRank:
    def test_basic_descending(self) -> None:
        values = np.array([30.0, 10.0, 50.0, 20.0, 40.0])
        ranks = competition_rank(values)
        # 50→1, 40→2, 30→3, 20→4, 10→5
        expected = [3, 5, 1, 4, 2]
        assert list(ranks) == expected

    def test_ties_share_rank(self) -> None:
        values = np.array([50.0, 30.0, 50.0, 10.0])
        ranks = competition_rank(values)
        # Two 50s tie for rank 1, 30→3 (skips 2), 10→4
        assert list(ranks) == [1, 3, 1, 4]

    def test_all_tied(self) -> None:
        values = np.array([42.0, 42.0, 42.0])
        ranks = competition_rank(values)
        assert list(ranks) == [1, 1, 1]

    def test_single_value(self) -> None:
        values = np.array([100.0])
        ranks = competition_rank(values)
        assert list(ranks) == [1]

    def test_empty_array(self) -> None:
        ranks = competition_rank(np.array([]))
        assert ranks.size == 0

    def test_already_sorted(self) -> None:
        values = np.array([100.0, 90.0, 80.0, 70.0])
        ranks = competition_rank(values)
        assert list(ranks) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Test: rank_metric
# ---------------------------------------------------------------------------


class TestRankMetric:
    def test_client_at_top(self) -> None:
        participants = {"client": 100.0, "comp_a": 80.0, "comp_b": 60.0}
        result = rank_metric("mrr", participants, "client")
        assert isinstance(result, MetricRank)
        assert result.rank == 1
        assert result.percentile == 100.0
        assert result.total_participants == 3

    def test_client_at_bottom(self) -> None:
        participants = {"client": 10.0, "comp_a": 80.0, "comp_b": 60.0}
        result = rank_metric("mrr", participants, "client")
        assert result.rank == 3
        assert result.percentile == 0.0

    def test_client_in_middle(self) -> None:
        participants = {"client": 50.0, "comp_a": 80.0, "comp_b": 20.0}
        result = rank_metric("mrr", participants, "client")
        assert result.rank == 2
        assert result.percentile == 50.0

    def test_client_tied_for_first(self) -> None:
        participants = {"client": 80.0, "comp_a": 80.0, "comp_b": 60.0}
        result = rank_metric("mrr", participants, "client")
        assert result.rank == 1
        assert result.percentile == 100.0

    def test_field_statistics(self) -> None:
        participants = {"client": 10.0, "comp_a": 20.0, "comp_b": 30.0}
        result = rank_metric("mrr", participants, "client")
        assert result.field_mean == 20.0
        assert result.field_median == 20.0
        assert result.client_value == 10.0

    def test_missing_client_raises(self) -> None:
        participants = {"comp_a": 80.0, "comp_b": 60.0}
        with pytest.raises(KeyError, match="not_here"):
            rank_metric("mrr", participants, "not_here")

    def test_single_participant(self) -> None:
        result = rank_metric("mrr", {"client": 42.0}, "client")
        assert result.rank == 1
        assert result.percentile == 100.0
        assert result.total_participants == 1


# ---------------------------------------------------------------------------
# Test: rank_competitive (integration)
# ---------------------------------------------------------------------------


_CLIENT_METRICS = {
    "mrr": 35_000.0,
    "growth_rate": 18.0,
    "churn_rate": 5.0,
}

_COMPETITORS = {
    "comp_a": {"mrr": 50_000.0, "growth_rate": 12.0, "churn_rate": 8.0},
    "comp_b": {"mrr": 20_000.0, "growth_rate": 25.0, "churn_rate": 3.0},
    "comp_c": {"mrr": 40_000.0, "growth_rate": 10.0, "churn_rate": 10.0},
}


class TestCompetitiveRanking:
    def test_returns_competitive_ranking(self) -> None:
        result = rank_competitive("client", _CLIENT_METRICS, _COMPETITORS)
        assert isinstance(result, CompetitiveRanking)

    def test_total_participants(self) -> None:
        result = rank_competitive("client", _CLIENT_METRICS, _COMPETITORS)
        assert result.total_participants == 4  # client + 3 competitors

    def test_overall_rank_bounded(self) -> None:
        result = rank_competitive("client", _CLIENT_METRICS, _COMPETITORS)
        assert 1 <= result.overall_rank <= 4

    def test_overall_percentile_bounded(self) -> None:
        result = rank_competitive("client", _CLIENT_METRICS, _COMPETITORS)
        assert 0.0 <= result.overall_percentile <= 100.0

    def test_tier_valid(self) -> None:
        result = rank_competitive("client", _CLIENT_METRICS, _COMPETITORS)
        assert result.tier in ("leader", "strong", "average", "weak")

    def test_metric_ranks_populated(self) -> None:
        result = rank_competitive("client", _CLIENT_METRICS, _COMPETITORS)
        assert "mrr" in result.metric_ranks
        assert "growth_rate" in result.metric_ranks
        assert "churn_rate" in result.metric_ranks

    def test_peer_scores_populated(self) -> None:
        result = rank_competitive("client", _CLIENT_METRICS, _COMPETITORS)
        assert "client" in result.peer_scores
        assert "comp_a" in result.peer_scores
        assert "comp_b" in result.peer_scores
        assert "comp_c" in result.peer_scores
        for score in result.peer_scores.values():
            assert 0.0 <= score <= 100.0

    def test_clear_leader_gets_top_tier(self) -> None:
        """Client beats all competitors on every metric → leader."""
        client = {"mrr": 100_000.0, "growth_rate": 50.0, "churn_rate": 30.0}
        comps = {
            "comp_a": {"mrr": 10_000.0, "growth_rate": 5.0, "churn_rate": 2.0},
            "comp_b": {"mrr": 20_000.0, "growth_rate": 8.0, "churn_rate": 3.0},
        }
        result = rank_competitive("client", client, comps)
        assert result.overall_rank == 1
        assert result.tier == "leader"

    def test_clear_last_gets_weak_tier(self) -> None:
        """Client loses on every metric → weak."""
        client = {"mrr": 1_000.0, "growth_rate": 1.0}
        comps = {
            "comp_a": {"mrr": 50_000.0, "growth_rate": 20.0},
            "comp_b": {"mrr": 40_000.0, "growth_rate": 15.0},
            "comp_c": {"mrr": 30_000.0, "growth_rate": 10.0},
            "comp_d": {"mrr": 20_000.0, "growth_rate": 8.0},
        }
        result = rank_competitive("client", client, comps)
        assert result.overall_rank == 5
        assert result.tier == "weak"

    def test_no_competitors(self) -> None:
        """Solo client should be rank 1, leader."""
        result = rank_competitive("client", {"mrr": 42.0}, {})
        assert result.overall_rank == 1
        assert result.total_participants == 1
        assert result.tier == "leader"
        assert result.overall_percentile == 100.0

    def test_partial_metric_overlap(self) -> None:
        """Competitors may not have all metrics — only shared metrics ranked."""
        client = {"mrr": 30_000.0, "exotic_kpi": 99.0}
        comps = {"comp_a": {"mrr": 40_000.0}}  # no exotic_kpi
        result = rank_competitive("client", client, comps)
        assert "mrr" in result.metric_ranks
        # exotic_kpi: only client has it → still ranked (solo = rank 1)
        assert "exotic_kpi" in result.metric_ranks
        assert result.metric_ranks["exotic_kpi"].rank == 1


# ---------------------------------------------------------------------------
# Test: Tie handling (integration)
# ---------------------------------------------------------------------------


class TestTieHandling:
    def test_overall_tie(self) -> None:
        """Two participants with identical metrics should tie."""
        client = {"mrr": 50_000.0, "growth_rate": 20.0}
        comps = {"comp_a": {"mrr": 50_000.0, "growth_rate": 20.0}}
        result = rank_competitive("client", client, comps)
        # Both should be rank 1
        assert result.overall_rank == 1
        assert result.peer_scores["client"] == result.peer_scores["comp_a"]

    def test_metric_tie_skips_rank(self) -> None:
        """Standard competition: two rank-1 ties → next is rank 3."""
        participants = {"client": 90.0, "comp_a": 90.0, "comp_b": 70.0}
        result = rank_metric("mrr", participants, "client")
        assert result.rank == 1
        # comp_b should be rank 3, not 2
        result_b = rank_metric("mrr", participants, "comp_b")
        assert result_b.rank == 3


# ---------------------------------------------------------------------------
# Test: Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_identical_output_on_repeat(self) -> None:
        r1 = rank_competitive("client", _CLIENT_METRICS, _COMPETITORS)
        r2 = rank_competitive("client", _CLIENT_METRICS, _COMPETITORS)
        assert r1.model_dump() == r2.model_dump()
