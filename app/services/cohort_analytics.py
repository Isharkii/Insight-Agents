from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


_BUSINESS_RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "business_rules.yaml"
DEFAULT_COHORT_KEYS: tuple[str, ...] = ("signup_month", "acquisition_channel", "segment")


@lru_cache(maxsize=1)
def _load_business_rules() -> dict[str, Any]:
    try:
        raw = _BUSINESS_RULES_PATH.read_text(encoding="utf-8")
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except (OSError, TypeError, ValueError):
        return {}


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: object, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


_COHORT_RULES = _as_dict(_load_business_rules().get("cohort_analytics"))
_COHORT_THRESHOLDS = _as_dict(_COHORT_RULES.get("thresholds"))


@dataclass(frozen=True)
class CohortAnalyticsConfig:
    min_points_per_cohort: int = _as_int(_COHORT_RULES.get("min_points_per_cohort"), 3)
    sparse_confidence_penalty: float = _as_float(
        _COHORT_RULES.get("sparse_confidence_penalty"),
        0.25,
    )
    missing_key_penalty: float = _as_float(
        _COHORT_RULES.get("missing_key_penalty"),
        0.10,
    )
    min_confidence: float = _as_float(_COHORT_RULES.get("min_confidence"), 0.20)
    severe_decay_threshold: float = _as_float(
        _COHORT_THRESHOLDS.get("severe_decay_threshold"),
        0.10,
    )
    moderate_decay_threshold: float = _as_float(
        _COHORT_THRESHOLDS.get("moderate_decay_threshold"),
        0.05,
    )
    severe_acceleration_threshold: float = _as_float(
        _COHORT_THRESHOLDS.get("severe_acceleration_threshold"),
        0.05,
    )
    moderate_acceleration_threshold: float = _as_float(
        _COHORT_THRESHOLDS.get("moderate_acceleration_threshold"),
        0.02,
    )
    zero_guard: float = _as_float(_COHORT_RULES.get("zero_guard"), 1e-9)


def compute_cohort_analytics(
    rows: Sequence[Mapping[str, Any]],
    *,
    cohort_keys: Sequence[str] = DEFAULT_COHORT_KEYS,
    active_metric_names: Sequence[str] = (),
    churn_metric_names: Sequence[str] = (),
    config: CohortAnalyticsConfig | None = None,
) -> dict[str, Any]:
    cfg = config or CohortAnalyticsConfig()
    normalized_rows = _normalize_rows(rows)
    if not normalized_rows:
        return {
            "status": "partial",
            "confidence_score": max(cfg.min_confidence, 1.0 - cfg.sparse_confidence_penalty),
            "warnings": ["No cohort-capable rows available."],
            "cohort_keys": list(cohort_keys),
            "cohorts_by_key": {},
            "signals": {
                "retention_decay": None,
                "lifetime_estimate": None,
                "churn_acceleration": None,
                "worst_cohort": None,
                "sparse_cohorts": 0,
                "risk_hint": "low",
            },
        }

    normalized_active = {_normalize(v) for v in active_metric_names if _normalize(v)}
    normalized_churn = {_normalize(v) for v in churn_metric_names if _normalize(v)}

    warnings: list[str] = []
    sparse_count = 0
    missing_key_count = 0
    cohorts_by_key: dict[str, Any] = {}
    all_decay: list[float] = []
    all_lifetime: list[float] = []
    all_acceleration: list[float] = []
    worst_cohort: dict[str, Any] | None = None

    for cohort_key in cohort_keys:
        key_name = str(cohort_key).strip()
        if not key_name:
            continue

        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in normalized_rows:
            cohort_value = _cohort_value(row, key_name)
            if not cohort_value:
                continue
            grouped.setdefault(cohort_value, []).append(row)

        if not grouped:
            missing_key_count += 1
            warnings.append(f"No rows found for cohort key '{key_name}'.")
            continue

        cohort_items: list[dict[str, Any]] = []
        for cohort_value, cohort_rows in sorted(grouped.items(), key=lambda item: item[0]):
            analysis = _analyze_one_cohort(
                cohort_rows,
                cohort_key=key_name,
                cohort_value=cohort_value,
                active_metric_names=normalized_active,
                churn_metric_names=normalized_churn,
                config=cfg,
            )
            cohort_items.append(analysis)

            decay = analysis.get("decay")
            lifetime = analysis.get("lifetime_estimate")
            acceleration = analysis.get("churn_acceleration")
            if isinstance(decay, (int, float)) and math.isfinite(decay):
                all_decay.append(float(decay))
            if isinstance(lifetime, (int, float)) and math.isfinite(lifetime):
                all_lifetime.append(float(lifetime))
            if isinstance(acceleration, (int, float)) and math.isfinite(acceleration):
                all_acceleration.append(float(acceleration))
            if analysis.get("status") == "partial":
                sparse_count += 1
                for message in analysis.get("warnings", []):
                    warnings.append(f"{key_name}={cohort_value}: {message}")

            worst_cohort = _select_worse_cohort(worst_cohort, analysis)

        cohorts_by_key[key_name] = {
            "cohorts": cohort_items,
            "count": len(cohort_items),
        }

    confidence = 1.0
    confidence -= float(sparse_count) * cfg.sparse_confidence_penalty
    confidence -= float(missing_key_count) * cfg.missing_key_penalty
    confidence = max(cfg.min_confidence, min(1.0, confidence))

    status = "partial" if (warnings or sparse_count > 0 or missing_key_count > 0) else "success"
    retention_decay = _safe_mean(all_decay)
    lifetime_estimate = _safe_mean(all_lifetime)
    churn_acceleration = _safe_mean(all_acceleration)
    risk_hint = _risk_hint(
        decay=retention_decay,
        acceleration=churn_acceleration,
        cfg=cfg,
    )
    public_worst = _public_worst_cohort(worst_cohort)

    return {
        "status": status,
        "confidence_score": round(confidence, 6),
        "warnings": warnings,
        "cohort_keys": [str(key) for key in cohort_keys],
        "cohorts_by_key": cohorts_by_key,
        "signals": {
            "retention_decay": _round_or_none(retention_decay),
            "lifetime_estimate": _round_or_none(lifetime_estimate),
            "churn_acceleration": _round_or_none(churn_acceleration),
            "worst_cohort": public_worst,
            "sparse_cohorts": sparse_count,
            "risk_hint": risk_hint,
        },
    }


def _normalize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        metadata = row.get("metadata_json")
        if not isinstance(metadata, Mapping):
            metadata = {}
        normalized.append(
            {
                "timestamp": _timestamp_text(row),
                "metric_name": str(row.get("metric_name") or "").strip(),
                "metric_value": _coerce_float(row.get("metric_value")),
                "signup_month": _cohort_field(row, metadata, "signup_month"),
                "acquisition_channel": _cohort_field(row, metadata, "acquisition_channel"),
                "segment": _cohort_field(row, metadata, "segment"),
                "metadata_json": dict(metadata),
            }
        )
    return [row for row in normalized if row["timestamp"]]


def _cohort_field(row: Mapping[str, Any], metadata: Mapping[str, Any], field: str) -> str | None:
    value = row.get(field)
    if value is None:
        value = metadata.get(field)
    if field == "acquisition_channel" and value is None:
        value = metadata.get("channel")
    if field == "segment" and value is None:
        value = metadata.get("customer_segment")
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _timestamp_text(row: Mapping[str, Any]) -> str:
    for key in ("timestamp", "period_end", "period_start", "created_at"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _cohort_value(row: Mapping[str, Any], cohort_key: str) -> str | None:
    value = row.get(cohort_key)
    if value is None:
        metadata = row.get("metadata_json")
        if isinstance(metadata, Mapping):
            value = metadata.get(cohort_key)
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _analyze_one_cohort(
    cohort_rows: Sequence[Mapping[str, Any]],
    *,
    cohort_key: str,
    cohort_value: str,
    active_metric_names: set[str],
    churn_metric_names: set[str],
    config: CohortAnalyticsConfig,
) -> dict[str, Any]:
    periods = _build_period_buckets(
        cohort_rows,
        active_metric_names=active_metric_names,
        churn_metric_names=churn_metric_names,
        zero_guard=config.zero_guard,
    )
    retention_curve: list[dict[str, Any]] = []
    retention_values: list[float] = []
    churn_rates: list[float] = []

    for age, period in enumerate(periods):
        active = period.get("active_customers")
        churn = period.get("churned_customers")
        retention = period.get("retention_rate")
        if isinstance(retention, (int, float)) and math.isfinite(retention):
            retention_values.append(float(retention))
        if isinstance(churn, (int, float)) and isinstance(active, (int, float)):
            churn_rates.append(float(churn) / max(float(active), config.zero_guard))

        retention_curve.append(
            {
                "age": age,
                "timestamp": period.get("timestamp"),
                "active_customers": _round_or_none(active),
                "churned_customers": _round_or_none(churn),
                "retention_rate": _round_or_none(retention),
            }
        )

    decay = _retention_decay(retention_values)
    lifetime_estimate = _lifetime_estimate(retention_values)
    churn_acceleration = _churn_acceleration(churn_rates)

    cohort_warnings: list[str] = []
    status = "success"
    confidence = 1.0
    if len(retention_curve) < config.min_points_per_cohort:
        status = "partial"
        confidence = max(config.min_confidence, 1.0 - config.sparse_confidence_penalty)
        cohort_warnings.append(
            f"Sparse cohort history: {len(retention_curve)} points "
            f"(minimum {config.min_points_per_cohort})."
        )

    return {
        "cohort_key": cohort_key,
        "cohort_value": cohort_value,
        "status": status,
        "confidence_score": round(confidence, 6),
        "warnings": cohort_warnings,
        "retention_curve": retention_curve,
        "decay": _round_or_none(decay),
        "lifetime_estimate": _round_or_none(lifetime_estimate),
        "churn_acceleration": _round_or_none(churn_acceleration),
    }


def _build_period_buckets(
    rows: Sequence[Mapping[str, Any]],
    *,
    active_metric_names: set[str],
    churn_metric_names: set[str],
    zero_guard: float,
) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, float | str | None]] = {}
    for row in rows:
        ts = str(row.get("timestamp") or "").strip()
        if not ts:
            continue
        metric_name = _normalize(row.get("metric_name"))
        metric_value = _coerce_float(row.get("metric_value"))
        if metric_value is None:
            continue
        bucket = buckets.setdefault(
            ts,
            {
                "timestamp": ts,
                "active_customers": 0.0,
                "churned_customers": 0.0,
                "retention_rate": None,
            },
        )
        kind = _metric_kind(
            metric_name,
            active_metric_names=active_metric_names,
            churn_metric_names=churn_metric_names,
        )
        if kind == "active":
            bucket["active_customers"] = float(bucket.get("active_customers") or 0.0) + metric_value
        elif kind == "churn":
            bucket["churned_customers"] = float(bucket.get("churned_customers") or 0.0) + metric_value
        elif kind == "retention":
            bucket["retention_rate"] = _normalize_rate(metric_value)

    ordered = [buckets[key] for key in sorted(buckets)]
    if not ordered:
        return []

    initial_active: float | None = None
    for bucket in ordered:
        active = _coerce_float(bucket.get("active_customers"))
        if active is not None and active > zero_guard:
            initial_active = active
            break

    for bucket in ordered:
        retention = _coerce_float(bucket.get("retention_rate"))
        if retention is None:
            active = _coerce_float(bucket.get("active_customers"))
            if (
                initial_active is not None
                and active is not None
                and initial_active > zero_guard
            ):
                retention = max(0.0, min(1.0, active / initial_active))
                bucket["retention_rate"] = retention
    return ordered


def _metric_kind(
    metric_name: str,
    *,
    active_metric_names: set[str],
    churn_metric_names: set[str],
) -> str | None:
    if metric_name in active_metric_names:
        return "active"
    if metric_name in churn_metric_names:
        return "churn"

    if "retention" in metric_name:
        return "retention"

    if "churn" in metric_name or "lost" in metric_name:
        return "churn"

    active_hints = ("active_customer", "active_user", "active_account", "active_subscriber")
    if any(hint in metric_name for hint in active_hints):
        return "active"

    return None


def _retention_decay(retention_values: Sequence[float]) -> float | None:
    if len(retention_values) < 2:
        return None
    drops = [
        max(0.0, retention_values[idx - 1] - retention_values[idx])
        for idx in range(1, len(retention_values))
    ]
    if not drops:
        return None
    return mean(drops)


def _lifetime_estimate(retention_values: Sequence[float]) -> float | None:
    if not retention_values:
        return None
    clipped = [max(0.0, min(1.0, value)) for value in retention_values]
    return sum(clipped)


def _churn_acceleration(churn_rates: Sequence[float]) -> float | None:
    if len(churn_rates) < 3:
        return None
    second_diff = [
        churn_rates[idx] - (2.0 * churn_rates[idx - 1]) + churn_rates[idx - 2]
        for idx in range(2, len(churn_rates))
    ]
    if not second_diff:
        return None
    return mean(second_diff)


def _safe_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return mean(values)


def _risk_hint(*, decay: float | None, acceleration: float | None, cfg: CohortAnalyticsConfig) -> str:
    decay_value = float(decay or 0.0)
    accel_value = float(acceleration or 0.0)
    if decay is not None and decay_value >= cfg.severe_decay_threshold:
        return "high"
    if acceleration is not None and accel_value >= cfg.severe_acceleration_threshold:
        return "high"
    if decay is not None and decay_value >= cfg.moderate_decay_threshold:
        return "moderate"
    if acceleration is not None and accel_value >= cfg.moderate_acceleration_threshold:
        return "moderate"
    return "low"


def _select_worse_cohort(
    current: dict[str, Any] | None,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    decay = _coerce_float(candidate.get("decay")) or 0.0
    acceleration = _coerce_float(candidate.get("churn_acceleration")) or 0.0
    score = decay + max(0.0, acceleration)
    payload = {
        "cohort_key": candidate.get("cohort_key"),
        "cohort_value": candidate.get("cohort_value"),
        "decay": _round_or_none(candidate.get("decay")),
        "churn_acceleration": _round_or_none(candidate.get("churn_acceleration")),
        "status": candidate.get("status"),
        "confidence_score": _round_or_none(candidate.get("confidence_score")),
        "_score": score,
    }
    if current is None:
        return payload
    current_score = _coerce_float(current.get("_score")) or 0.0
    if score > current_score:
        return payload
    return current


def _normalize_rate(value: float | None) -> float | None:
    if value is None:
        return None
    if abs(value) > 1.0 and abs(value) <= 100.0:
        return value / 100.0
    return value


def _public_worst_cohort(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "cohort_key": value.get("cohort_key"),
        "cohort_value": value.get("cohort_value"),
        "decay": _round_or_none(value.get("decay")),
        "churn_acceleration": _round_or_none(value.get("churn_acceleration")),
        "status": value.get("status"),
        "confidence_score": _round_or_none(value.get("confidence_score")),
    }


def _round_or_none(value: Any) -> float | None:
    numeric = _coerce_float(value)
    if numeric is None:
        return None
    return round(numeric, 6)


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()
