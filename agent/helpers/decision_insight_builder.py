"""
agent/helpers/decision_insight_builder.py

Decision-Grade Insight Layer: compresses raw pipeline signals into a
high-signal, LLM-optimized structured digest.

Design principles:
    1. Every field is deterministic — no LLM, no side-effects.
    2. Missing data is explicitly represented (None / "unavailable"),
       never defaulted to 0.0.
    3. Every component carries its own confidence score.
    4. Output is a clean JSON-serializable dict — no raw blobs.
    5. O(n) or O(n log n) — no heavy dependencies.

This module replaces the previous pattern of passing raw node payloads
through the prompt builder with truncation.  Instead, it produces a
compressed, structured insight digest that the LLM can reason from
directly without needing to parse nested envelopes.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_TREND_THRESHOLD = 0.05  # ±5% half-over-half change → accelerating/declining
_VOLATILITY_LOW = 0.10   # CV thresholds
_VOLATILITY_HIGH = 0.30
_ANOMALY_ZSCORE = 2.5    # z-score threshold for anomaly flagging
_MIN_SERIES_FOR_DELTA = 2
_MIN_SERIES_FOR_TREND = 3
_MIN_SERIES_FOR_ANOMALY = 5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KPIInsight:
    """Structured insight for a single KPI metric."""
    metric: str
    status: str           # "growing" | "declining" | "stable" | "unknown"
    latest_value: float | None = None
    previous_value: float | None = None
    change_pct: float | None = None
    trend_strength: float | None = None  # slope magnitude
    volatility: float | None = None      # coefficient of variation
    volatility_label: str = "unknown"    # "low" | "medium" | "high"
    confidence: float = 0.0
    data_points: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "metric": self.metric,
            "status": self.status,
            "latest_value": _round_safe(self.latest_value),
            "previous_value": _round_safe(self.previous_value),
            "change_pct": _round_safe(self.change_pct),
            "trend_strength": _round_safe(self.trend_strength),
            "volatility": _round_safe(self.volatility),
            "volatility_label": self.volatility_label,
            "confidence": round(self.confidence, 4),
            "data_points": self.data_points,
        }
        if self.warnings:
            d["warnings"] = self.warnings
        return d


@dataclass
class AnomalyRecord:
    """A single detected anomaly."""
    metric: str
    index: int
    value: float
    z_score: float
    anomaly_type: str  # "spike" | "dip" | "level_shift"
    magnitude: float   # absolute deviation from mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "index": self.index,
            "value": _round_safe(self.value),
            "z_score": round(self.z_score, 3),
            "type": self.anomaly_type,
            "magnitude": _round_safe(self.magnitude),
        }


@dataclass
class DeltaRecord:
    """Period-over-period change for a metric."""
    metric: str
    period_label: str   # "MoM" | "QoQ" | "WoW"
    absolute_change: float | None = None
    pct_change: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "period": self.period_label,
            "absolute_change": _round_safe(self.absolute_change),
            "pct_change": _round_safe(self.pct_change),
        }


@dataclass
class DriverRecord:
    """A contributor to observed change."""
    name: str
    contribution_value: float | None = None
    contribution_pct: float | None = None
    direction: str = "unknown"   # "positive" | "negative" | "neutral"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "contribution_value": _round_safe(self.contribution_value),
            "contribution_pct": _round_safe(self.contribution_pct),
            "direction": self.direction,
        }


@dataclass
class ForecastSummary:
    """Compressed forecast for LLM consumption."""
    metric: str
    direction: str            # "upward" | "downward" | "flat" | "unknown"
    slope: float | None = None
    r_squared: float | None = None
    confidence: float = 0.0
    is_valid: bool = False
    validity_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "direction": self.direction,
            "slope": _round_safe(self.slope),
            "r_squared": _round_safe(self.r_squared),
            "confidence": round(self.confidence, 4),
            "is_valid": self.is_valid,
            "validity_reason": self.validity_reason or None,
        }


@dataclass
class DataQualityComponent:
    """Per-component data quality assessment."""
    component: str
    status: str           # "available" | "partial" | "unavailable"
    confidence: float
    data_points: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "component": self.component,
            "status": self.status,
            "confidence": round(self.confidence, 4),
            "data_points": self.data_points,
        }
        if self.warnings:
            d["warnings"] = self.warnings
        return d


# ---------------------------------------------------------------------------
# Core computation functions (all O(n) or O(n log n))
# ---------------------------------------------------------------------------


def compute_kpi_insights(
    metric_series: Mapping[str, Sequence[float | Any]],
) -> list[KPIInsight]:
    """Compute structured KPI insights for all metrics in the series."""
    insights: list[KPIInsight] = []

    for name in sorted(metric_series):
        raw_values = metric_series[name]
        values = _coerce_series(raw_values)
        n = len(values)

        if n == 0:
            insights.append(KPIInsight(
                metric=name,
                status="unknown",
                warnings=["No valid data points"],
            ))
            continue

        latest = values[-1]
        previous = values[-2] if n >= 2 else None
        change_pct = _safe_pct_change(previous, latest)
        status = _classify_status(change_pct)
        trend_strength = _compute_trend_strength(values) if n >= _MIN_SERIES_FOR_TREND else None
        volatility = _compute_cv(values)
        vol_label = _classify_volatility(volatility)

        # Confidence: based on data depth and quality
        depth_score = min(1.0, n / 12.0)  # saturates at 12 points
        vol_penalty = min(0.3, (volatility or 0.0) * 0.3) if volatility else 0.0
        confidence = max(0.1, depth_score - vol_penalty)

        warnings: list[str] = []
        if n < _MIN_SERIES_FOR_TREND:
            warnings.append(f"Only {n} data points — trend unreliable")
        if vol_label == "high":
            warnings.append(f"High volatility (CV={volatility:.3f})")

        insights.append(KPIInsight(
            metric=name,
            status=status,
            latest_value=latest,
            previous_value=previous,
            change_pct=change_pct,
            trend_strength=trend_strength,
            volatility=volatility,
            volatility_label=vol_label,
            confidence=round(confidence, 4),
            data_points=n,
            warnings=warnings,
        ))

    return insights


def detect_anomalies_zscore(
    metric_series: Mapping[str, Sequence[float | Any]],
    *,
    threshold: float = _ANOMALY_ZSCORE,
) -> list[AnomalyRecord]:
    """Detect statistical outliers using z-score method. O(n) per metric."""
    anomalies: list[AnomalyRecord] = []

    for name in sorted(metric_series):
        values = _coerce_series(metric_series[name])
        n = len(values)
        if n < _MIN_SERIES_FOR_ANOMALY:
            continue

        mean_val = sum(values) / n
        variance = sum((v - mean_val) ** 2 for v in values) / n
        std_dev = math.sqrt(max(variance, 1e-12))

        for idx, val in enumerate(values):
            z = (val - mean_val) / std_dev
            if abs(z) >= threshold:
                anomaly_type = "spike" if z > 0 else "dip"
                anomalies.append(AnomalyRecord(
                    metric=name,
                    index=idx,
                    value=val,
                    z_score=z,
                    anomaly_type=anomaly_type,
                    magnitude=abs(val - mean_val),
                ))

    return anomalies


def compute_deltas(
    metric_series: Mapping[str, Sequence[float | Any]],
) -> list[DeltaRecord]:
    """Compute period-over-period deltas for each metric. O(n)."""
    deltas: list[DeltaRecord] = []

    for name in sorted(metric_series):
        values = _coerce_series(metric_series[name])
        n = len(values)

        if n >= 2:
            # Most recent period change (MoM equivalent)
            prev, curr = values[-2], values[-1]
            deltas.append(DeltaRecord(
                metric=name,
                period_label="MoM",
                absolute_change=curr - prev,
                pct_change=_safe_pct_change(prev, curr),
            ))

        if n >= 4:
            # Quarter-over-quarter (3-period lookback)
            prev_q, curr_q = values[-4], values[-1]
            deltas.append(DeltaRecord(
                metric=name,
                period_label="QoQ",
                absolute_change=curr_q - prev_q,
                pct_change=_safe_pct_change(prev_q, curr_q),
            ))

    return deltas


def extract_drivers(
    root_cause_payload: Mapping[str, Any] | None,
    role_contribution_payload: Mapping[str, Any] | None = None,
) -> list[DriverRecord]:
    """Extract ranked drivers from root cause and role contribution data."""
    drivers: list[DriverRecord] = []

    # From root cause analysis
    if root_cause_payload:
        contributing = root_cause_payload.get("contributing_factors")
        if isinstance(contributing, list):
            for factor in contributing:
                if isinstance(factor, str):
                    drivers.append(DriverRecord(name=factor, direction="negative"))
                elif isinstance(factor, dict):
                    drivers.append(DriverRecord(
                        name=str(factor.get("name") or factor.get("factor", "unknown")),
                        contribution_value=_safe_float(factor.get("impact")),
                        direction=str(factor.get("direction", "negative")),
                    ))

        root_causes = root_cause_payload.get("root_causes")
        if isinstance(root_causes, list):
            for cause in root_causes:
                if isinstance(cause, str) and not any(d.name == cause for d in drivers):
                    drivers.append(DriverRecord(name=cause, direction="negative"))

    # From role/segment contribution
    if role_contribution_payload:
        top_contributors = role_contribution_payload.get("top_contributors")
        if isinstance(top_contributors, list):
            for contrib in top_contributors:
                if isinstance(contrib, dict):
                    name = str(contrib.get("name", "unknown"))
                    value = _safe_float(contrib.get("contribution_value"))
                    drivers.append(DriverRecord(
                        name=name,
                        contribution_value=value,
                        contribution_pct=_safe_float(contrib.get("contribution_pct")),
                        direction="positive" if (value or 0) > 0 else "negative",
                    ))

    return drivers


def summarize_forecasts(
    forecast_payload: Mapping[str, Any] | None,
) -> list[ForecastSummary]:
    """Compress forecast data into LLM-ready summaries."""
    summaries: list[ForecastSummary] = []

    if not forecast_payload:
        return summaries

    forecasts = forecast_payload.get("forecasts")
    if not isinstance(forecasts, dict):
        return summaries

    for metric_name, row in sorted(forecasts.items()):
        if not isinstance(row, dict):
            continue
        data = row.get("forecast_data")
        if not isinstance(data, dict):
            continue

        slope = _safe_float(data.get("slope"))
        r_squared = _safe_float(data.get("r_squared"))
        if r_squared is None:
            regression = data.get("regression")
            if isinstance(regression, dict):
                r_squared = _safe_float(regression.get("r_squared"))

        conf = _safe_float(data.get("confidence_score")) or 0.0

        # Direction
        if slope is None:
            direction = "unknown"
        elif slope > 0.01:
            direction = "upward"
        elif slope < -0.01:
            direction = "downward"
        else:
            direction = "flat"

        # Validity
        is_valid = True
        validity_reason = ""
        if r_squared is not None and r_squared < 0.2:
            is_valid = False
            validity_reason = f"R²={r_squared:.3f} too low (< 0.2)"
        elif r_squared is not None and r_squared < 0.4:
            validity_reason = f"R²={r_squared:.3f} moderate — use with caution"

        dp_used = data.get("datapoints_used")
        if isinstance(dp_used, int) and dp_used < 4:
            is_valid = False
            validity_reason = f"Only {dp_used} datapoints — insufficient for reliable forecast"

        summaries.append(ForecastSummary(
            metric=str(metric_name),
            direction=direction,
            slope=slope,
            r_squared=r_squared,
            confidence=conf,
            is_valid=is_valid,
            validity_reason=validity_reason,
        ))

    return summaries


def assess_data_quality(
    *,
    kpi_status: str,
    kpi_confidence: float,
    kpi_data_points: int,
    forecast_status: str,
    forecast_confidence: float,
    cohort_status: str,
    cohort_confidence: float,
    risk_status: str,
    risk_confidence: float,
    growth_status: str,
    growth_confidence: float,
) -> tuple[list[DataQualityComponent], list[str]]:
    """Assess per-component data quality and identify missing components."""
    components: list[DataQualityComponent] = []
    missing: list[str] = []

    for name, status, conf, dp in [
        ("kpi", kpi_status, kpi_confidence, kpi_data_points),
        ("forecast", forecast_status, forecast_confidence, 0),
        ("cohort", cohort_status, cohort_confidence, 0),
        ("risk", risk_status, risk_confidence, 0),
        ("growth", growth_status, growth_confidence, 0),
    ]:
        if status in ("failed", "skipped"):
            components.append(DataQualityComponent(
                component=name, status="unavailable", confidence=0.0,
                data_points=dp,
                warnings=[f"{name} data is {status}"],
            ))
            missing.append(name)
        elif status == "insufficient_data":
            components.append(DataQualityComponent(
                component=name, status="partial", confidence=conf,
                data_points=dp,
                warnings=[f"{name} has insufficient data"],
            ))
        else:
            components.append(DataQualityComponent(
                component=name,
                status="available",
                confidence=conf,
                data_points=dp,
            ))

    return components, missing


# ---------------------------------------------------------------------------
# Main builder: assembles full insight digest
# ---------------------------------------------------------------------------


def build_insight_digest(
    *,
    metric_series: Mapping[str, Sequence[float | Any]],
    forecast_payload: Mapping[str, Any] | None = None,
    root_cause_payload: Mapping[str, Any] | None = None,
    role_contribution_payload: Mapping[str, Any] | None = None,
    risk_payload: Mapping[str, Any] | None = None,
    growth_payload: Mapping[str, Any] | None = None,
    cohort_payload: Mapping[str, Any] | None = None,
    kpi_status: str = "success",
    kpi_confidence: float = 1.0,
    forecast_status: str = "success",
    forecast_confidence: float = 1.0,
    cohort_status: str = "skipped",
    cohort_confidence: float = 0.0,
    risk_status: str = "success",
    risk_confidence: float = 1.0,
    growth_status: str = "success",
    growth_confidence: float = 1.0,
    entity_name: str = "",
) -> dict[str, Any]:
    """Build the complete Decision-Grade Insight Digest.

    Returns a fully structured, JSON-serializable dict that replaces
    raw node payloads in the LLM prompt.  Every field is computed
    deterministically.  Missing data is represented as None, never 0.0.
    """
    # 1. KPI Insights
    kpi_insights = compute_kpi_insights(metric_series)
    kpi_data_points = sum(k.data_points for k in kpi_insights)

    # 2. Anomalies
    anomalies = detect_anomalies_zscore(metric_series)

    # 3. Deltas
    deltas = compute_deltas(metric_series)

    # 4. Drivers
    drivers = extract_drivers(root_cause_payload, role_contribution_payload)

    # 5. Forecast summaries
    forecast_summaries = summarize_forecasts(forecast_payload)

    # 6. Data quality
    quality_components, missing_components = assess_data_quality(
        kpi_status=kpi_status,
        kpi_confidence=kpi_confidence,
        kpi_data_points=kpi_data_points,
        forecast_status=forecast_status,
        forecast_confidence=forecast_confidence,
        cohort_status=cohort_status,
        cohort_confidence=cohort_confidence,
        risk_status=risk_status,
        risk_confidence=risk_confidence,
        growth_status=growth_status,
        growth_confidence=growth_confidence,
    )

    # 7. Overall confidence: weighted by signal authority
    authority_weights = {
        "kpi": 1.0,
        "growth": 0.8,
        "risk": 0.6,
        "forecast": 0.4,
        "cohort": 0.3,
    }
    weighted_sum = 0.0
    weight_total = 0.0
    for comp in quality_components:
        w = authority_weights.get(comp.component, 0.3)
        if comp.status != "unavailable":
            weighted_sum += comp.confidence * w
            weight_total += w

    overall_confidence = round(
        weighted_sum / weight_total if weight_total > 0 else 0.0, 4
    )

    # 8. Risk summary extraction
    risk_summary = _extract_risk_summary(risk_payload)

    # 9. Growth summary extraction
    growth_summary = _extract_growth_summary(growth_payload)

    digest: dict[str, Any] = {
        "entity_name": entity_name or None,
        "overall_confidence": overall_confidence,
        "kpi_insights": [k.to_dict() for k in kpi_insights],
        "drivers": [d.to_dict() for d in drivers] if drivers else [],
        "anomalies": [a.to_dict() for a in anomalies] if anomalies else [],
        "deltas": [d.to_dict() for d in deltas] if deltas else [],
        "forecast_summary": [f.to_dict() for f in forecast_summaries] if forecast_summaries else [],
        "risk_summary": risk_summary,
        "growth_summary": growth_summary,
        "data_quality": {
            "overall_confidence": overall_confidence,
            "components": [c.to_dict() for c in quality_components],
            "missing_components": missing_components,
        },
    }

    logger.info(
        "Insight digest built: entity=%s confidence=%.3f kpi_count=%d "
        "anomaly_count=%d driver_count=%d forecast_count=%d missing=%s",
        entity_name,
        overall_confidence,
        len(kpi_insights),
        len(anomalies),
        len(drivers),
        len(forecast_summaries),
        missing_components,
    )

    return digest


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_series(values: Sequence[Any]) -> list[float]:
    """Convert a sequence to a clean float list, skipping non-finite values."""
    out: list[float] = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _safe_float(value: Any) -> float | None:
    """Convert to float, returning None on failure."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _safe_pct_change(previous: float | None, current: float | None) -> float | None:
    """Compute percentage change, returning None if not possible."""
    if previous is None or current is None:
        return None
    if previous == 0.0:
        return None
    return (current - previous) / abs(previous)


def _classify_status(change_pct: float | None) -> str:
    """Classify metric status from percentage change."""
    if change_pct is None:
        return "unknown"
    if change_pct > _TREND_THRESHOLD:
        return "growing"
    if change_pct < -_TREND_THRESHOLD:
        return "declining"
    return "stable"


def _compute_trend_strength(values: list[float]) -> float | None:
    """Compute linear regression slope as trend strength indicator. O(n)."""
    n = len(values)
    if n < 2:
        return None

    # Simple linear regression: slope = Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²)
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n

    numerator = 0.0
    denominator = 0.0
    for i, y in enumerate(values):
        x_dev = i - x_mean
        numerator += x_dev * (y - y_mean)
        denominator += x_dev * x_dev

    if denominator < 1e-12:
        return 0.0

    slope = numerator / denominator
    # Normalize by mean to get relative slope
    if abs(y_mean) > 1e-12:
        return slope / abs(y_mean)
    return slope


def _compute_cv(values: list[float]) -> float | None:
    """Compute coefficient of variation. O(n)."""
    n = len(values)
    if n < 2:
        return None
    mean_val = sum(values) / n
    if abs(mean_val) < 1e-12:
        return 0.0
    variance = sum((v - mean_val) ** 2 for v in values) / n
    return math.sqrt(variance) / abs(mean_val)


def _classify_volatility(cv: float | None) -> str:
    """Classify volatility from coefficient of variation."""
    if cv is None:
        return "unknown"
    if cv > _VOLATILITY_HIGH:
        return "high"
    if cv > _VOLATILITY_LOW:
        return "medium"
    return "low"


def _round_safe(value: float | None, digits: int = 4) -> float | None:
    """Round a value, returning None if input is None."""
    if value is None:
        return None
    return round(value, digits)


def _extract_risk_summary(risk_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract a clean risk summary from the risk node payload."""
    if not risk_payload:
        return {"status": "unavailable"}

    summary: dict[str, Any] = {"status": "available"}

    risk_score = _safe_float(risk_payload.get("risk_score"))
    if risk_score is not None:
        summary["risk_score"] = round(risk_score, 2)

    risk_level = risk_payload.get("risk_level")
    if isinstance(risk_level, str):
        summary["risk_level"] = risk_level

    categories = risk_payload.get("risk_categories")
    if isinstance(categories, list):
        summary["risk_categories"] = [
            {
                "category": str(cat.get("category") or cat.get("name", "unknown")),
                "severity": str(cat.get("severity", "unknown")),
                "score": _round_safe(_safe_float(cat.get("score"))),
            }
            for cat in categories
            if isinstance(cat, dict)
        ]

    return summary


def _extract_growth_summary(growth_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract a clean growth summary from the growth node payload."""
    if not growth_payload:
        return {"status": "unavailable"}

    summary: dict[str, Any] = {"status": "available"}

    primary_metric = growth_payload.get("primary_metric")
    if primary_metric:
        summary["primary_metric"] = str(primary_metric)

    horizons = growth_payload.get("primary_horizons")
    if isinstance(horizons, dict):
        for key in ("short_growth", "mid_growth", "long_growth", "cagr", "trend_acceleration"):
            val = _safe_float(horizons.get(key))
            if val is not None:
                summary[key] = round(val, 4)

    momentum = growth_payload.get("momentum")
    if momentum is not None:
        summary["momentum"] = str(momentum) if isinstance(momentum, str) else _round_safe(_safe_float(momentum))

    return summary
