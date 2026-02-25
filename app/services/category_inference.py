"""
app/services/category_inference.py

Deterministic, confidence-scored business category inference engine.

Uses three weighted signals to infer business category from dataset characteristics:
1. Metric-based inference (weight 0.5) — presence of category-specific metric names
2. Column semantic inference (weight 0.3) — fuzzy matching of column names to category aliases
3. Data shape inference (weight 0.2) — structural patterns like time-series, segments, transactions

No LLM usage. Fully deterministic. Same input always produces same output.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz import fuzz

from app.services.category_registry import (
    CategoryPack,
    get_category_pack,
    supported_categories,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal definitions — core metrics strongly associated with each category
# ---------------------------------------------------------------------------

_CATEGORY_CORE_METRICS: dict[str, tuple[str, ...]] = {
    "saas": (
        "mrr", "arr", "churn_rate", "ltv", "cac",
        "recurring_revenue", "active_subscriptions", "churned_subscriptions",
        "arpu", "expansion_revenue", "contraction_revenue",
    ),
    "ecommerce": (
        "aov", "cart_abandonment", "roas", "conversion_rate",
        "gmv", "orders", "purchase_frequency", "cac",
        "revenue", "active_customers", "new_customers",
    ),
    "financial_markets": (
        "close_price", "volume", "drawdown", "volatility",
        "sharpe_like", "aum_revenue", "net_trading_income",
        "active_accounts", "risk_free_rate", "pnl",
    ),
    "marketing_analytics": (
        "impressions", "clicks", "ctr", "cpc",
        "ad_spend", "roas", "conversions", "campaign_revenue",
        "attributed_revenue", "pipeline_dropoffs",
    ),
    "healthcare": (
        "readmission_rate", "occupancy_rate", "bed_occupancy_rate",
        "patient_revenue", "active_patients", "readmissions",
        "occupied_beds", "total_beds", "staff_hours", "care_revenue",
    ),
    "agency": (
        "billable_hours", "utilization", "utilization_rate",
        "team_performance", "project_revenue", "billable_rate",
        "non_billable_hours", "client_satisfaction",
    ),
    "operations": (
        "throughput", "defect_rate", "uptime_rate",
        "output_value", "labor_hours", "downtime_hours",
        "capacity_hours", "failed_units", "processed_units",
    ),
    "retail": (
        "net_sales", "footfall", "average_ticket", "gross_margin_rate",
        "pos_sales", "active_shoppers", "cogs",
        "conversion_rate", "retail_revenue",
    ),
    "general_timeseries": (
        "revenue", "value", "count", "total",
    ),
}

# Column name patterns that suggest specific data shapes
_TIME_COLUMNS: frozenset[str] = frozenset({
    "date", "timestamp", "time", "period", "month", "year", "quarter",
    "week", "day", "created_at", "updated_at", "report_date",
})

_SEGMENT_COLUMNS: frozenset[str] = frozenset({
    "role", "segment", "department", "team", "region", "channel",
    "tier", "plan", "cohort", "group", "division",
})

_TRANSACTION_COLUMNS: frozenset[str] = frozenset({
    "order_id", "transaction_id", "invoice_id", "payment_id",
    "cart_id", "checkout_id", "sku", "product_id", "item_id",
})

# Weights for each signal
_WEIGHT_METRIC = 0.5
_WEIGHT_SEMANTIC = 0.3
_WEIGHT_SHAPE = 0.2

# Thresholds
_THRESHOLD_SUCCESS = 0.80
_THRESHOLD_AMBIGUOUS = 0.60

# Fuzzy match threshold (rapidfuzz score 0-100)
_FUZZY_THRESHOLD = 75

# Fallback category
_FALLBACK_CATEGORY = "general_timeseries"

# Cross-category disambiguation: minimum gap between #1 and #2 to avoid penalty
_DISAMBIGUATION_GAP_THRESHOLD = 0.15
# Maximum penalty applied when #1 and #2 are nearly tied
_DISAMBIGUATION_MAX_PENALTY = 0.15


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class CategoryInferenceResult(BaseModel):
    """Result of deterministic category inference."""

    model_config = ConfigDict(frozen=True)

    inferred_category: str | None = None
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    alternative_candidates: list[dict[str, Any]] = Field(default_factory=list)
    status: str = "insufficient_data"  # success | ambiguous | insufficient_data


# ---------------------------------------------------------------------------
# Signal scorers
# ---------------------------------------------------------------------------


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _score_metric_signal(
    metric_names: list[str],
    core_metrics: tuple[str, ...],
) -> tuple[float, list[str]]:
    """Score based on overlap between dataset metric names and category core metrics.

    Uses Jaccard-like scoring: matched / min(core_count, metric_count).
    Applies a coverage boost when >= 3 core metrics match, as this is strong
    evidence of category affinity even if the dataset has many other metrics.

    Returns (score, evidence_list).
    """
    if not core_metrics:
        return 0.0, []

    normalized_metrics = {_normalize(m) for m in metric_names if m}
    if not normalized_metrics:
        return 0.0, []

    matched: list[str] = []
    for core in core_metrics:
        core_norm = _normalize(core)
        if core_norm in normalized_metrics:
            matched.append(core)
            continue
        # Check partial containment
        for metric in normalized_metrics:
            if core_norm in metric or metric in core_norm:
                matched.append(f"{core}~={metric}")
                break

    n_matched = len(matched)
    # Score relative to the smaller set (precision-oriented)
    denominator = min(len(core_metrics), len(normalized_metrics))
    base_score = n_matched / denominator if denominator > 0 else 0.0

    # Boost: having >= 3 core matches is strong categorical evidence
    if n_matched >= 5:
        base_score = min(1.0, base_score * 1.3)
    elif n_matched >= 3:
        base_score = min(1.0, base_score * 1.15)

    evidence = [f"metric_match: {m}" for m in matched] if matched else []
    return min(1.0, base_score), evidence


def _score_semantic_signal(
    column_names: list[str],
    metric_names: list[str],
    category: str,
) -> tuple[float, list[str]]:
    """Score based on fuzzy similarity between dataset terms and category alias packs.

    Considers both column headers and metric names as dataset identifiers.
    Uses metric_aliases, category_aliases, optional_signals, and rate_metrics
    from the category registry.
    Returns (score, evidence_list).
    """
    pack = get_category_pack(category)
    if pack is None:
        return 0.0, []

    # Build target terms from pack aliases and core metrics
    target_terms: set[str] = set()
    for key, aliases in pack.metric_aliases.items():
        target_terms.add(_normalize(key))
        for alias in aliases:
            target_terms.add(_normalize(alias))
    for alias in pack.category_aliases:
        target_terms.add(_normalize(alias))
    for signal in pack.optional_signals:
        target_terms.add(_normalize(signal))
    for rate in pack.rate_metrics:
        target_terms.add(_normalize(rate))
    # Include core metrics for this category (broader signal vocabulary)
    for core in _CATEGORY_CORE_METRICS.get(category, ()):
        target_terms.add(_normalize(core))

    if not target_terms:
        return 0.0, []

    # Combine column names and metric names as dataset identifiers
    dataset_terms: set[str] = set()
    for c in column_names:
        if c:
            dataset_terms.add(_normalize(c))
    for m in metric_names:
        if m:
            dataset_terms.add(_normalize(m))

    if not dataset_terms:
        return 0.0, []

    matched_count = 0
    evidence: list[str] = []

    # Score: what fraction of category-specific terms appear in dataset terms
    for target in sorted(target_terms):
        best_score = 0.0
        best_term = ""
        for term in dataset_terms:
            if term == target:
                best_score = 100.0
                best_term = term
                break
            ratio = fuzz.ratio(term, target)
            if ratio > best_score:
                best_score = ratio
                best_term = term

        if best_score >= _FUZZY_THRESHOLD:
            matched_count += 1
            if best_score == 100.0:
                evidence.append(f"semantic_exact: {best_term}={target}")
            else:
                evidence.append(f"semantic_fuzzy: {best_term}~{target}({best_score:.0f}%)")

    score = matched_count / len(target_terms) if target_terms else 0.0
    return min(1.0, score), evidence


def _score_shape_signal(
    column_names: list[str],
    row_count: int,
    unique_entity_count: int,
    category: str,
) -> tuple[float, list[str]]:
    """Score based on structural data shape characteristics.

    Considers: time-series presence, segment/role dimensions, transaction-level patterns.
    Returns (score, evidence_list).
    """
    normalized_columns = {_normalize(c) for c in column_names if c}
    evidence: list[str] = []
    sub_scores: list[float] = []

    # Time-series presence (common across most categories)
    time_cols = normalized_columns & _TIME_COLUMNS
    has_time = bool(time_cols)
    if has_time:
        evidence.append(f"time_columns: {sorted(time_cols)}")

    # Segment/role dimensions
    segment_cols = normalized_columns & _SEGMENT_COLUMNS
    has_segments = bool(segment_cols)
    if has_segments:
        evidence.append(f"segment_columns: {sorted(segment_cols)}")

    # Transaction-level indicators
    transaction_cols = normalized_columns & _TRANSACTION_COLUMNS
    has_transactions = bool(transaction_cols)
    if has_transactions:
        evidence.append(f"transaction_columns: {sorted(transaction_cols)}")

    # Category-specific shape scoring
    if category in ("saas", "financial_markets", "general_timeseries"):
        # These categories strongly expect time-series data
        sub_scores.append(0.8 if has_time else 0.1)
        sub_scores.append(0.5 if has_segments else 0.3)
        sub_scores.append(0.2 if not has_transactions else 0.3)
    elif category in ("ecommerce", "retail"):
        # Can be transaction-level or aggregated
        sub_scores.append(0.5 if has_time else 0.2)
        sub_scores.append(0.3 if has_segments else 0.2)
        sub_scores.append(0.7 if has_transactions else 0.3)
    elif category == "healthcare":
        sub_scores.append(0.6 if has_time else 0.2)
        sub_scores.append(0.6 if has_segments else 0.2)
        sub_scores.append(0.2 if not has_transactions else 0.3)
    elif category in ("marketing_analytics",):
        sub_scores.append(0.5 if has_time else 0.2)
        sub_scores.append(0.5 if has_segments else 0.2)
        sub_scores.append(0.3 if not has_transactions else 0.4)
    elif category in ("agency",):
        sub_scores.append(0.5 if has_time else 0.2)
        sub_scores.append(0.7 if has_segments else 0.2)
        sub_scores.append(0.2 if not has_transactions else 0.3)
    elif category in ("operations",):
        sub_scores.append(0.6 if has_time else 0.2)
        sub_scores.append(0.4 if has_segments else 0.2)
        sub_scores.append(0.3 if not has_transactions else 0.4)
    else:
        # Generic fallback shape scoring
        sub_scores.append(0.4 if has_time else 0.2)
        sub_scores.append(0.3 if has_segments else 0.2)
        sub_scores.append(0.3 if has_transactions else 0.2)

    # Row count signal — more rows suggest richer data (diminishing returns)
    if row_count > 100:
        sub_scores.append(0.5)
    elif row_count > 20:
        sub_scores.append(0.3)
    else:
        sub_scores.append(0.1)

    score = sum(sub_scores) / len(sub_scores) if sub_scores else 0.0
    return min(1.0, score), evidence


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------


def infer_category(
    *,
    metric_names: list[str],
    column_names: list[str],
    row_count: int = 0,
    unique_entity_count: int = 1,
) -> CategoryInferenceResult:
    """Infer business category from dataset characteristics.

    Args:
        metric_names: Distinct metric_name values found in the dataset.
        column_names: CSV column headers (post-mapping or raw).
        row_count: Total row count in the dataset.
        unique_entity_count: Number of distinct entities.

    Returns:
        CategoryInferenceResult with inferred category, confidence, and evidence.
    """
    candidates: list[dict[str, Any]] = []
    categories = list(supported_categories())

    # Also score against core metrics dict entries not in registry
    for cat_name in _CATEGORY_CORE_METRICS:
        if cat_name not in categories:
            categories.append(cat_name)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_categories: list[str] = []
    for c in categories:
        if c not in seen:
            seen.add(c)
            unique_categories.append(c)

    for category in unique_categories:
        core_metrics = _CATEGORY_CORE_METRICS.get(category, ())

        metric_score, metric_evidence = _score_metric_signal(metric_names, core_metrics)
        semantic_score, semantic_evidence = _score_semantic_signal(
            column_names, metric_names, category,
        )
        shape_score, shape_evidence = _score_shape_signal(
            column_names, row_count, unique_entity_count, category,
        )

        # Adaptive weighting: when metric evidence is strong, let it dominate
        if metric_score >= 0.8:
            w_metric, w_semantic, w_shape = 0.70, 0.15, 0.15
        else:
            w_metric, w_semantic, w_shape = (
                _WEIGHT_METRIC, _WEIGHT_SEMANTIC, _WEIGHT_SHAPE,
            )

        confidence = (
            w_metric * metric_score
            + w_semantic * semantic_score
            + w_shape * shape_score
        )
        confidence = round(min(1.0, max(0.0, confidence)), 4)

        all_evidence = metric_evidence + semantic_evidence + shape_evidence
        candidates.append({
            "category": category,
            "confidence": confidence,
            "metric_score": round(metric_score, 4),
            "semantic_score": round(semantic_score, 4),
            "shape_score": round(shape_score, 4),
            "evidence": all_evidence,
        })

    # Sort by confidence descending, then alphabetically for ties
    candidates.sort(key=lambda c: (-c["confidence"], c["category"]))

    if not candidates:
        return CategoryInferenceResult(
            inferred_category=None,
            confidence_score=0.0,
            evidence=["no_categories_available"],
            alternative_candidates=[],
            status="insufficient_data",
        )

    best = candidates[0]
    best_confidence: float = best["confidence"]
    best_category: str = best["category"]

    # Cross-category disambiguation penalty:
    # When the top two candidates are close in score, it signals genuine
    # ambiguity. Apply a proportional penalty so that near-ties don't
    # produce false high-confidence results.
    if len(candidates) >= 2:
        runner_up_confidence: float = candidates[1]["confidence"]
        gap = best_confidence - runner_up_confidence
        if gap < _DISAMBIGUATION_GAP_THRESHOLD and runner_up_confidence > 0.0:
            # Penalty scales linearly: zero gap → full penalty, threshold gap → zero
            penalty_ratio = 1.0 - (gap / _DISAMBIGUATION_GAP_THRESHOLD)
            penalty = _DISAMBIGUATION_MAX_PENALTY * penalty_ratio
            best_confidence = round(max(0.0, best_confidence - penalty), 4)
            best["confidence"] = best_confidence
            best["evidence"].append(
                f"disambiguation_penalty: -{penalty:.4f} "
                f"(gap={gap:.4f}, runner_up={candidates[1]['category']})"
            )

    # Decision rules
    if best_confidence >= _THRESHOLD_SUCCESS:
        status = "success"
        inferred_category = best_category
    elif best_confidence >= _THRESHOLD_AMBIGUOUS:
        status = "ambiguous"
        inferred_category = best_category
    else:
        status = "insufficient_data"
        inferred_category = None

    # Build alternative candidates (exclude the best)
    alternatives = [
        {
            "category": c["category"],
            "confidence": c["confidence"],
            "metric_score": c["metric_score"],
            "semantic_score": c["semantic_score"],
            "shape_score": c["shape_score"],
        }
        for c in candidates[1:]
        if c["confidence"] > 0.0
    ]

    return CategoryInferenceResult(
        inferred_category=inferred_category,
        confidence_score=best_confidence,
        evidence=best["evidence"],
        alternative_candidates=alternatives,
        status=status,
    )


# ---------------------------------------------------------------------------
# Validation helper — compare inferred vs user-supplied category
# ---------------------------------------------------------------------------


def validate_category_match(
    *,
    user_category: str | None,
    inference_result: CategoryInferenceResult,
) -> list[str]:
    """Compare user-supplied category against inferred result.

    Returns a list of warning messages (empty if no issues).
    """
    warnings: list[str] = []
    if not user_category:
        return warnings

    user_norm = _normalize(user_category)

    if inference_result.status == "success" and inference_result.inferred_category:
        inferred_norm = _normalize(inference_result.inferred_category)
        if user_norm != inferred_norm:
            # Check if user category is an alias of the inferred category
            pack = get_category_pack(inferred_norm)
            user_is_alias = False
            if pack:
                user_is_alias = user_norm in {_normalize(a) for a in pack.category_aliases}

            if not user_is_alias:
                warnings.append(
                    f"User-supplied category '{user_category}' differs from "
                    f"inferred category '{inference_result.inferred_category}' "
                    f"(confidence={inference_result.confidence_score:.2f}). "
                    f"Review dataset metrics for accuracy."
                )

    return warnings
