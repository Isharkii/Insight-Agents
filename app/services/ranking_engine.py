"""
app/services/ranking_engine.py

Competitive ranking engine — ranks a client relative to named competitors
within an industry on a per-metric and overall basis.

All computations are deterministic, NumPy-vectorised, and LLM-free.

Ranking method
--------------
Standard competition ranking ("1224"):  for each value *v*,
    rank(v) = 1 + count(all values > v)

Ties receive the same rank and the next rank is skipped accordingly.

Competitive percentile
----------------------
    percentile = (N − rank) / (N − 1) × 100

where N is the total number of participants.  When N = 1 the percentile
is defined as 100 (sole participant is the leader by default).

Tier classification
-------------------
    percentile ≥ 75  →  leader
    percentile ≥ 50  →  strong
    percentile ≥ 25  →  average
    percentile <  25 →  weak
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class MetricRank(BaseModel):
    """Ranking result for a single metric across all participants."""

    model_config = ConfigDict(frozen=True)

    metric_name: str
    client_value: float
    rank: int = Field(
        ...,
        ge=1,
        description=(
            "1-based rank (1 = best). Uses standard competition ranking — "
            "ties share the same rank and the next rank is skipped."
        ),
    )
    total_participants: int = Field(..., ge=1)
    percentile: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Competitive percentile: (N − rank) / (N − 1) × 100.",
    )
    field_mean: float = Field(..., description="Mean value across all participants.")
    field_median: float = Field(..., description="Median value across all participants.")


class CompetitiveRanking(BaseModel):
    """Overall competitive ranking of a client among peers."""

    model_config = ConfigDict(frozen=True)

    overall_rank: int = Field(..., ge=1, description="Rank by mean percentile across metrics.")
    total_participants: int = Field(..., ge=1)
    overall_percentile: float = Field(..., ge=0.0, le=100.0)
    tier: str = Field(
        ...,
        description="Competitive tier: leader, strong, average, or weak.",
    )
    metric_ranks: dict[str, MetricRank] = Field(
        default_factory=dict,
        description="Per-metric ranking breakdown.",
    )
    peer_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Mean percentile per participant (for transparency).",
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TIER_THRESHOLDS: list[tuple[float, str]] = [
    (75.0, "leader"),
    (50.0, "strong"),
    (25.0, "average"),
]
_TIER_DEFAULT = "weak"


# ---------------------------------------------------------------------------
# Ranking primitives
# ---------------------------------------------------------------------------


def _classify_tier(percentile: float) -> str:
    """Map a competitive percentile to a tier label.

    Thresholds
    ----------
    ≥ 75  →  leader
    ≥ 50  →  strong
    ≥ 25  →  average
    <  25 →  weak
    """
    for threshold, label in _TIER_THRESHOLDS:
        if percentile >= threshold:
            return label
    return _TIER_DEFAULT


def _competitive_percentile(rank: int, n: int) -> float:
    """Compute competitive percentile from rank and participant count.

    Formula
    -------
    percentile = (N − rank) / (N − 1) × 100

    When N = 1 the sole participant is assigned percentile 100.
    """
    if n <= 1:
        return 100.0
    return round((n - rank) / (n - 1) * 100.0, 4)


def competition_rank(values: np.ndarray) -> np.ndarray:
    """Standard competition ranking (descending — higher value = better rank).

    For each element *v* in *values*:

        rank(v) = 1 + count(values > v)

    This produces the "1224" pattern: ties share the minimum rank and
    the next rank is skipped.

    Parameters
    ----------
    values:
        1-D array of finite floats.

    Returns
    -------
    np.ndarray
        Integer ranks (1-based), same shape as *values*.
    """
    if values.size == 0:
        return np.array([], dtype=int)
    # Broadcasting: count how many values are strictly greater than each value
    ranks = np.sum(values[np.newaxis, :] > values[:, np.newaxis], axis=1) + 1
    return ranks.astype(int)


# ---------------------------------------------------------------------------
# Per-metric ranking
# ---------------------------------------------------------------------------


def rank_metric(
    metric_name: str,
    participants: Mapping[str, float],
    client_name: str,
) -> MetricRank:
    """Rank the client on a single metric against all participants.

    Parameters
    ----------
    metric_name:
        Human-readable metric identifier.
    participants:
        ``{participant_name: metric_value}`` — must include the client.
    client_name:
        Key identifying the client within *participants*.

    Returns
    -------
    MetricRank
        The client's rank, percentile, and field statistics.

    Raises
    ------
    KeyError
        If *client_name* is not found in *participants*.
    """
    if client_name not in participants:
        raise KeyError(f"Client '{client_name}' not found in participants")

    names = list(participants.keys())
    values = np.array([float(participants[n]) for n in names], dtype=float)

    ranks = competition_rank(values)
    client_idx = names.index(client_name)
    client_rank = int(ranks[client_idx])
    n = len(names)

    return MetricRank(
        metric_name=metric_name,
        client_value=round(float(values[client_idx]), 6),
        rank=client_rank,
        total_participants=n,
        percentile=_competitive_percentile(client_rank, n),
        field_mean=round(float(np.mean(values)), 6),
        field_median=round(float(np.median(values)), 6),
    )


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def rank_competitive(
    client_name: str,
    client_metrics: dict[str, float],
    competitor_metrics: dict[str, dict[str, float]],
) -> CompetitiveRanking:
    """Rank the client relative to competitors across all shared metrics.

    Parameters
    ----------
    client_name:
        Identifier for the client.
    client_metrics:
        ``{metric_name: value}`` for the client.
    competitor_metrics:
        ``{competitor_name: {metric_name: value}}`` for each competitor.

    Returns
    -------
    CompetitiveRanking
        Overall rank, tier, per-metric breakdown, and peer score
        transparency.

    Algorithm
    ---------
    1. Merge client into participant pool.
    2. For each metric where the client has a value, collect all
       participants who also have that metric and rank them.
    3. Compute each participant's mean percentile across all metrics
       they participated in.
    4. Rank participants by mean percentile to determine overall rank.
    5. Classify the client into a tier based on overall percentile.
    """
    # 1. Build participant pool
    all_names: list[str] = [client_name] + list(competitor_metrics.keys())
    n_total = len(all_names)

    # 2. Per-metric ranking (only metrics the client has)
    metric_ranks: dict[str, MetricRank] = {}
    # Track per-participant percentiles across metrics
    participant_percentiles: dict[str, list[float]] = {name: [] for name in all_names}

    for metric_name, client_val in client_metrics.items():
        try:
            cv = float(client_val)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(cv):
            continue

        # Collect participants who have this metric
        metric_participants: dict[str, float] = {client_name: cv}
        for comp_name, comp_metrics in competitor_metrics.items():
            if metric_name in comp_metrics:
                try:
                    comp_val = float(comp_metrics[metric_name])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(comp_val):
                    metric_participants[comp_name] = comp_val

        if len(metric_participants) < 1:
            continue

        # Rank this metric
        mr = rank_metric(metric_name, metric_participants, client_name)
        metric_ranks[metric_name] = mr

        # Compute percentiles for all participants in this metric
        p_names = list(metric_participants.keys())
        p_values = np.array([metric_participants[n] for n in p_names], dtype=float)
        p_ranks = competition_rank(p_values)
        n_p = len(p_names)
        for idx, name in enumerate(p_names):
            pct = _competitive_percentile(int(p_ranks[idx]), n_p)
            participant_percentiles[name].append(pct)

    # 3. Mean percentile per participant
    peer_scores: dict[str, float] = {}
    for name in all_names:
        pcts = participant_percentiles[name]
        if pcts:
            peer_scores[name] = round(float(np.mean(pcts)), 2)
        else:
            peer_scores[name] = 0.0

    # 4. Overall rank: rank participants by mean percentile (descending)
    score_names = list(peer_scores.keys())
    score_values = np.array([peer_scores[n] for n in score_names], dtype=float)
    overall_ranks = competition_rank(score_values)
    client_overall_idx = score_names.index(client_name)
    client_overall_rank = int(overall_ranks[client_overall_idx])

    # 5. Percentile and tier
    overall_pct = _competitive_percentile(client_overall_rank, n_total)
    tier = _classify_tier(overall_pct)

    return CompetitiveRanking(
        overall_rank=client_overall_rank,
        total_participants=n_total,
        overall_percentile=overall_pct,
        tier=tier,
        metric_ranks=metric_ranks,
        peer_scores=peer_scores,
    )
