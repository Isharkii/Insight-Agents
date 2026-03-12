"""
app/services/decision_engine_service.py

Synchronous decision intelligence service orchestration.

Pipeline:
    data -> insight -> reasoning -> strategy
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, Mapping, Sequence

from app.services.statistics.anomaly import detect_anomalies
from app.services.statistics.confidence_scoring import compute_confidence
from app.services.unit_economics import analyze_unit_economics
from forecast.robust_forecast import (
    MIN_POINTS_INSUFFICIENT,
    MIN_POINTS_REGRESSION,
    RobustForecast,
)

logger = logging.getLogger(__name__)


_FORECAST_HORIZON = 3
_MIN_RELIABLE_FORECAST_POINTS = MIN_POINTS_REGRESSION
_MIN_RELIABLE_FORECAST_CONFIDENCE = 0.30
_MIN_RELIABLE_FORECAST_CONFIDENCE_HIGH_VOL = 0.40
_MIN_RELIABLE_REGRESSION_R_SQUARED = 0.30

_DEFAULT_MODULE_TIMEOUTS_SECONDS: dict[str, float] = {
    "kpi_extraction": 0.30,
    "trend_analysis": 0.30,
    "anomaly_detection": 0.30,
    "forecasting": 2.00,
    "unit_economics": 1.00,
}

_VOLATILITY_CONFIDENCE_FACTOR = {
    "low": 1.00,
    "normal": 0.92,
    "high": 0.75,
    "insufficient_history": 0.80,
    "unknown": 0.85,
}

_FORECAST_WEIGHT_BY_VOLATILITY = {
    "low": 0.35,
    "normal": 0.35,
    "high": 0.20,
    "insufficient_history": 0.15,
    "unknown": 0.25,
}

def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _coerce_numeric_series(raw: Any) -> list[float]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    return [float(v) for v in raw if _is_numeric(v)]


def _clamp(value: float, *, lower: float = 0.0, upper: float = 0.99) -> float:
    return max(lower, min(upper, value))


def _slope(values: Sequence[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / float(n)
    num = 0.0
    den = 0.0
    for i, val in enumerate(values):
        dx = i - x_mean
        num += dx * (val - y_mean)
        den += dx * dx
    if den == 0.0:
        return 0.0
    return num / den


def _detect_anomaly(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"has_anomaly": False, "outlier_count": 0, "z_threshold": 2.5}

    mean = sum(values) / float(len(values))
    variance = sum((v - mean) ** 2 for v in values) / float(len(values))
    stdev = variance ** 0.5
    if stdev == 0.0:
        return {"has_anomaly": False, "outlier_count": 0, "z_threshold": 2.5}

    outlier_count = 0
    for val in values:
        z = abs((val - mean) / stdev)
        if z >= 2.5:
            outlier_count += 1

    return {
        "has_anomaly": outlier_count > 0,
        "outlier_count": outlier_count,
        "z_threshold": 2.5,
    }


def _trend_label(values: Sequence[float]) -> Literal["up", "down", "flat"]:
    s = _slope(values)
    if s > 0:
        return "up"
    if s < 0:
        return "down"
    return "flat"


@dataclass(frozen=True)
class BusinessInsight:
    signals: list[str]
    confidence_score: float
    kpis: dict[str, float]
    trend: dict[str, Any]
    anomaly: dict[str, Any]
    forecast: dict[str, Any]
    unit_economics: dict[str, Any]
    module_execution: dict[str, Any]
    confidence_breakdown: dict[str, Any]


@dataclass(frozen=True)
class AsyncModuleResult:
    module_name: str
    status: Literal["ok", "insufficient_data", "timeout", "error"]
    payload: dict[str, Any]
    execution_time_ms: float
    error: str | None = None


@dataclass(frozen=True)
class ForecastGuardrailDecision:
    accepted: bool
    status: Literal["accepted", "rejected"]
    reason: str
    reason_code: str
    input_points: int
    minimum_required: int
    tier: str
    volatility_regime: str
    raw_confidence_score: float
    adjusted_confidence_score: float
    required_confidence_score: float
    volatility_factor: float
    r_squared: float | None


class DecisionEngineService:
    """Synchronous decision intelligence orchestration service."""

    def __init__(
        self,
        *,
        module_timeouts: Mapping[str, float] | None = None,
    ) -> None:
        self._forecast_model = RobustForecast()
        self._module_timeouts = self._resolve_module_timeouts(module_timeouts)

    @staticmethod
    def _resolve_module_timeouts(
        module_timeouts: Mapping[str, float] | None,
    ) -> dict[str, float]:
        resolved = dict(_DEFAULT_MODULE_TIMEOUTS_SECONDS)
        if not isinstance(module_timeouts, Mapping):
            return resolved
        for raw_name, raw_timeout in module_timeouts.items():
            name = str(raw_name or "").strip().lower()
            if name not in resolved:
                continue
            try:
                parsed = float(raw_timeout)
            except (TypeError, ValueError):
                continue
            resolved[name] = max(0.001, parsed)
        return resolved

    def _module_timeout_seconds(self, module_name: str) -> float:
        return max(
            0.001,
            float(
                self._module_timeouts.get(
                    module_name,
                    _DEFAULT_MODULE_TIMEOUTS_SECONDS.get(module_name, 1.0),
                )
            ),
        )

    @staticmethod
    def _normalize_module_status(status: Any) -> Literal["ok", "insufficient_data"]:
        normalized = str(status or "").strip().lower()
        if normalized in {"insufficient_data", "insufficient-history"}:
            return "insufficient_data"
        return "ok"

    async def _run_module_with_timeout(
        self,
        *,
        module_name: str,
        fn: Callable[[], dict[str, Any]],
    ) -> AsyncModuleResult:
        timeout = self._module_timeout_seconds(module_name)
        started = perf_counter()
        try:
            raw_payload = await asyncio.wait_for(
                asyncio.to_thread(fn),
                timeout=timeout,
            )
            payload = dict(raw_payload) if isinstance(raw_payload, Mapping) else {}
            status = self._normalize_module_status(payload.get("status"))
            return AsyncModuleResult(
                module_name=module_name,
                status=status,
                payload=payload,
                execution_time_ms=round((perf_counter() - started) * 1000.0, 3),
            )
        except asyncio.TimeoutError:
            reason = (
                f"module '{module_name}' timed out after {timeout:.2f}s"
            )
            return AsyncModuleResult(
                module_name=module_name,
                status="timeout",
                payload={
                    "status": "timeout",
                    "reason": reason,
                },
                execution_time_ms=round((perf_counter() - started) * 1000.0, 3),
                error="timeout",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Async insight module failed module=%s error=%s", module_name, exc)
            return AsyncModuleResult(
                module_name=module_name,
                status="error",
                payload={
                    "status": "error",
                    "reason": str(exc),
                },
                execution_time_ms=round((perf_counter() - started) * 1000.0, 3),
                error=type(exc).__name__,
            )

    @staticmethod
    def _kpi_extraction_module(
        *,
        values: Sequence[float],
        churn_rate: float,
        conversion_rate: float,
        total_customers: float,
        revenue_latest: float,
    ) -> dict[str, Any]:
        kpis = {
            "revenue": round(revenue_latest, 6),
            "churn_rate": round(churn_rate, 6),
            "conversion_rate": round(conversion_rate, 6),
            "revenue_per_customer": round(revenue_latest / max(1.0, total_customers), 6),
        }
        return {
            "status": "ok" if values else "insufficient_data",
            "points": len(values),
            "kpis": kpis,
        }

    @staticmethod
    def _trend_analysis_module(values: Sequence[float]) -> dict[str, Any]:
        return {
            "status": "ok" if len(values) >= 2 else "insufficient_data",
            "direction": _trend_label(values),
            "slope": round(_slope(values), 6),
        }

    @staticmethod
    def _anomaly_detection_module(values: Sequence[float]) -> dict[str, Any]:
        if not values:
            return {
                "status": "insufficient_data",
                "has_anomaly": False,
                "outlier_count": 0,
                "method": "ensemble",
                "details": {
                    "status": "insufficient_history",
                    "anomaly_indexes": [],
                    "anomaly_values": [],
                },
            }
        details = detect_anomalies(values)
        anomaly_indexes = details.get("anomaly_indexes")
        if not isinstance(anomaly_indexes, list):
            anomaly_indexes = []
        return {
            "status": "ok" if len(values) >= 2 else "insufficient_data",
            "has_anomaly": bool(anomaly_indexes),
            "outlier_count": len(anomaly_indexes),
            "method": str(details.get("method") or "ensemble"),
            "details": details,
        }

    @staticmethod
    def _build_unit_economics_records(
        *,
        values: Sequence[float],
        churn_rate: float,
        conversion_rate: float,
        total_customers: float,
        context: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not values:
            return []
        ltv = _safe_float(context.get("ltv"), default=0.0) if _is_numeric(context.get("ltv")) else None
        cac = _safe_float(context.get("cac"), default=0.0) if _is_numeric(context.get("cac")) else None
        records: list[dict[str, Any]] = []
        for idx, revenue_value in enumerate(values):
            growth_rate = None
            if idx > 0:
                previous = values[idx - 1]
                if abs(previous) > 1e-9:
                    growth_rate = (revenue_value - previous) / abs(previous)
            computed: dict[str, Any] = {
                "revenue": {"value": revenue_value},
                "recurring_revenue": {"value": revenue_value},
                "mrr": {"value": revenue_value},
                "total_revenue": {"value": revenue_value},
                "churn_rate": {"value": churn_rate},
                "conversion_rate": {"value": conversion_rate},
                "active_customers": {"value": total_customers},
                "revenue_per_customer": {
                    "value": revenue_value / max(1.0, total_customers)
                },
            }
            if growth_rate is not None:
                computed["growth_rate"] = {"value": growth_rate}
            if ltv is not None:
                computed["ltv"] = {"value": ltv}
                computed["client_ltv"] = {"value": ltv}
            if cac is not None:
                computed["cac"] = {"value": cac}
            records.append(
                {
                    "period_start": f"2026-{idx + 1:02d}-01T00:00:00+00:00",
                    "period_end": f"2026-{idx + 1:02d}-28T00:00:00+00:00",
                    "created_at": f"2026-{idx + 1:02d}-28T00:00:00+00:00",
                    "computed_kpis": computed,
                }
            )
        return records

    @staticmethod
    def _unit_economics_module(
        *,
        values: Sequence[float],
        context: Mapping[str, Any],
        business_type: str,
        churn_rate: float,
        conversion_rate: float,
        total_customers: float,
    ) -> dict[str, Any]:
        records = DecisionEngineService._build_unit_economics_records(
            values=values,
            churn_rate=churn_rate,
            conversion_rate=conversion_rate,
            total_customers=total_customers,
            context=context,
        )
        if not records:
            return {
                "status": "insufficient_data",
                "signals": [],
                "warnings": ["No revenue history available for unit economics."],
            }
        payload = analyze_unit_economics(records, business_type=business_type)
        return dict(payload) if isinstance(payload, Mapping) else {
            "status": "insufficient_data",
            "signals": [],
            "warnings": ["Unit economics module returned an invalid payload."],
        }

    async def _execute_insight_modules_async(
        self,
        *,
        values: Sequence[float],
        payload: Mapping[str, Any],
        business_type: str,
        churn_rate: float,
        conversion_rate: float,
        total_customers: float,
        revenue_latest: float,
    ) -> dict[str, AsyncModuleResult]:
        tasks: dict[str, asyncio.Task[AsyncModuleResult]] = {
            "kpi_extraction": asyncio.create_task(
                self._run_module_with_timeout(
                    module_name="kpi_extraction",
                    fn=lambda: self._kpi_extraction_module(
                        values=values,
                        churn_rate=churn_rate,
                        conversion_rate=conversion_rate,
                        total_customers=total_customers,
                        revenue_latest=revenue_latest,
                    ),
                )
            ),
            "trend_analysis": asyncio.create_task(
                self._run_module_with_timeout(
                    module_name="trend_analysis",
                    fn=lambda: self._trend_analysis_module(values),
                )
            ),
            "anomaly_detection": asyncio.create_task(
                self._run_module_with_timeout(
                    module_name="anomaly_detection",
                    fn=lambda: self._anomaly_detection_module(values),
                )
            ),
            "forecasting": asyncio.create_task(
                self._run_module_with_timeout(
                    module_name="forecasting",
                    fn=lambda: self._safe_forecast(values),
                )
            ),
            "unit_economics": asyncio.create_task(
                self._run_module_with_timeout(
                    module_name="unit_economics",
                    fn=lambda: self._unit_economics_module(
                        values=values,
                        context=payload,
                        business_type=business_type,
                        churn_rate=churn_rate,
                        conversion_rate=conversion_rate,
                        total_customers=total_customers,
                    ),
                )
            ),
        }
        results = await asyncio.gather(*tasks.values())
        return {
            name: result
            for name, result in zip(tasks.keys(), results)
        }

    @staticmethod
    def _volatility_regime_from_forecast(forecast_result: Mapping[str, Any]) -> str:
        data_quality = forecast_result.get("data_quality")
        if isinstance(data_quality, Mapping):
            volatility = data_quality.get("volatility")
            if isinstance(volatility, Mapping):
                regime = str(volatility.get("regime") or "").strip().lower()
                if regime:
                    return regime
        return "unknown"

    @staticmethod
    def _guardrail_payload(decision: ForecastGuardrailDecision) -> dict[str, Any]:
        return {
            "status": decision.status,
            "accepted": decision.accepted,
            "reason": decision.reason,
            "reason_code": decision.reason_code,
            "input_points": decision.input_points,
            "minimum_required": decision.minimum_required,
            "tier": decision.tier,
            "volatility_regime": decision.volatility_regime,
            "raw_confidence_score": round(decision.raw_confidence_score, 6),
            "adjusted_confidence_score": round(decision.adjusted_confidence_score, 6),
            "required_confidence_score": round(decision.required_confidence_score, 6),
            "volatility_factor": round(decision.volatility_factor, 6),
            "r_squared": (
                round(decision.r_squared, 6)
                if decision.r_squared is not None
                else None
            ),
        }

    def _forecast_guardrail_decision(
        self,
        forecast_result: Mapping[str, Any],
    ) -> ForecastGuardrailDecision:
        status = str(forecast_result.get("status") or "").strip().lower()
        tier = str(forecast_result.get("tier") or "insufficient").strip().lower()
        input_points = int(
            forecast_result.get("input_points")
            or 0
        )
        minimum_required = int(
            forecast_result.get("minimum_required")
            or MIN_POINTS_INSUFFICIENT
        )
        confidence = _safe_float(forecast_result.get("confidence_score"), default=0.0)
        volatility_regime = self._volatility_regime_from_forecast(forecast_result)
        volatility_factor = _VOLATILITY_CONFIDENCE_FACTOR.get(
            volatility_regime,
            _VOLATILITY_CONFIDENCE_FACTOR["unknown"],
        )
        adjusted_confidence = _clamp(
            confidence * volatility_factor,
            lower=0.0,
            upper=0.99,
        )
        required_confidence = (
            _MIN_RELIABLE_FORECAST_CONFIDENCE_HIGH_VOL
            if volatility_regime == "high"
            else _MIN_RELIABLE_FORECAST_CONFIDENCE
        )

        regression = forecast_result.get("regression")
        r_squared: float | None = None
        if isinstance(regression, Mapping):
            raw_r2 = regression.get("r_squared")
            if raw_r2 is not None:
                r_squared = _safe_float(raw_r2, default=0.0)

        if status != "ok":
            return ForecastGuardrailDecision(
                accepted=False,
                status="rejected",
                reason="Forecast status is not ok.",
                reason_code="status_not_ok",
                input_points=input_points,
                minimum_required=minimum_required,
                tier=tier,
                volatility_regime=volatility_regime,
                raw_confidence_score=confidence,
                adjusted_confidence_score=0.0,
                required_confidence_score=required_confidence,
                volatility_factor=volatility_factor,
                r_squared=r_squared,
            )

        if input_points < _MIN_RELIABLE_FORECAST_POINTS:
            return ForecastGuardrailDecision(
                accepted=False,
                status="rejected",
                reason=(
                    f"Need at least {_MIN_RELIABLE_FORECAST_POINTS} points for a "
                    f"reliable forecast, got {input_points}."
                ),
                reason_code="below_reliable_datapoint_threshold",
                input_points=input_points,
                minimum_required=_MIN_RELIABLE_FORECAST_POINTS,
                tier=tier,
                volatility_regime=volatility_regime,
                raw_confidence_score=confidence,
                adjusted_confidence_score=adjusted_confidence,
                required_confidence_score=required_confidence,
                volatility_factor=volatility_factor,
                r_squared=r_squared,
            )

        if tier == "minimal":
            return ForecastGuardrailDecision(
                accepted=False,
                status="rejected",
                reason=(
                    "Minimal tier forecasts are not considered reliable for "
                    "decisioning."
                ),
                reason_code="minimal_tier_rejected",
                input_points=input_points,
                minimum_required=_MIN_RELIABLE_FORECAST_POINTS,
                tier=tier,
                volatility_regime=volatility_regime,
                raw_confidence_score=confidence,
                adjusted_confidence_score=adjusted_confidence,
                required_confidence_score=required_confidence,
                volatility_factor=volatility_factor,
                r_squared=r_squared,
            )

        if (
            r_squared is not None
            and r_squared < _MIN_RELIABLE_REGRESSION_R_SQUARED
        ):
            return ForecastGuardrailDecision(
                accepted=False,
                status="rejected",
                reason=(
                    f"Regression fit too weak (r_squared={r_squared:.3f} < "
                    f"{_MIN_RELIABLE_REGRESSION_R_SQUARED:.3f})."
                ),
                reason_code="poor_regression_fit",
                input_points=input_points,
                minimum_required=_MIN_RELIABLE_FORECAST_POINTS,
                tier=tier,
                volatility_regime=volatility_regime,
                raw_confidence_score=confidence,
                adjusted_confidence_score=adjusted_confidence,
                required_confidence_score=required_confidence,
                volatility_factor=volatility_factor,
                r_squared=r_squared,
            )

        if adjusted_confidence < required_confidence:
            return ForecastGuardrailDecision(
                accepted=False,
                status="rejected",
                reason=(
                    f"Volatility-adjusted confidence ({adjusted_confidence:.3f}) "
                    f"is below required threshold ({required_confidence:.3f})."
                ),
                reason_code="volatility_adjusted_low_confidence",
                input_points=input_points,
                minimum_required=_MIN_RELIABLE_FORECAST_POINTS,
                tier=tier,
                volatility_regime=volatility_regime,
                raw_confidence_score=confidence,
                adjusted_confidence_score=adjusted_confidence,
                required_confidence_score=required_confidence,
                volatility_factor=volatility_factor,
                r_squared=r_squared,
            )

        return ForecastGuardrailDecision(
            accepted=True,
            status="accepted",
            reason="Forecast passed guardrail checks.",
            reason_code="accepted",
            input_points=input_points,
            minimum_required=_MIN_RELIABLE_FORECAST_POINTS,
            tier=tier,
            volatility_regime=volatility_regime,
            raw_confidence_score=confidence,
            adjusted_confidence_score=adjusted_confidence,
            required_confidence_score=required_confidence,
            volatility_factor=volatility_factor,
            r_squared=r_squared,
        )

    def _safe_forecast(self, values: Sequence[float]) -> dict[str, Any]:
        raw = self._forecast_model.forecast(list(values))
        tier = str(raw.get("tier") or "insufficient")
        status = str(raw.get("status") or "insufficient_data")
        minimum_required = int(raw.get("minimum_required") or MIN_POINTS_INSUFFICIENT)
        input_points = int(raw.get("input_points") or len(values))
        guardrail = self._forecast_guardrail_decision(raw)

        raw_forecast = raw.get("forecast")
        projections = raw_forecast if isinstance(raw_forecast, Mapping) else {}
        values_list = [
            projections.get("month_1"),
            projections.get("month_2"),
            projections.get("month_3"),
        ]

        warnings = [str(item) for item in (raw.get("warnings") or [])]
        diagnostics = raw.get("diagnostics")
        volatility_regime = guardrail.volatility_regime

        if status == "insufficient_data":
            reason = ""
            if isinstance(diagnostics, Mapping):
                reason = str(diagnostics.get("reason") or "").strip()
            return {
                "status": "insufficient_data",
                "forecast_available": False,
                "horizon": _FORECAST_HORIZON,
                "values": [None, None, None],
                "projections": {"month_1": None, "month_2": None, "month_3": None},
                "tier": "insufficient",
                "confidence_score": 0.0,
                "raw_confidence_score": round(guardrail.raw_confidence_score, 6),
                "input_points": input_points,
                "minimum_required": minimum_required,
                "volatility_regime": "insufficient_history",
                "reason": reason or (
                    f"Insufficient data for forecasting: need at least "
                    f"{minimum_required} points, got {input_points}."
                ),
                "warnings": warnings,
                "diagnostics": diagnostics if isinstance(diagnostics, Mapping) else {},
                "guardrail": self._guardrail_payload(guardrail),
                "raw": raw,
            }

        if not guardrail.accepted:
            rejected_warnings = list(warnings)
            rejected_warnings.append(
                f"Forecast rejected by guardrail: {guardrail.reason_code}."
            )
            return {
                "status": "insufficient_data",
                "forecast_available": False,
                "horizon": _FORECAST_HORIZON,
                "values": [None, None, None],
                "projections": {"month_1": None, "month_2": None, "month_3": None},
                "tier": tier,
                "confidence_score": 0.0,
                "raw_confidence_score": round(guardrail.raw_confidence_score, 6),
                "input_points": input_points,
                "minimum_required": guardrail.minimum_required,
                "volatility_regime": volatility_regime or "unknown",
                "reason": guardrail.reason,
                "rejection_reason_code": guardrail.reason_code,
                "warnings": rejected_warnings,
                "diagnostics": diagnostics if isinstance(diagnostics, Mapping) else {},
                "guardrail": self._guardrail_payload(guardrail),
                "raw": raw,
            }

        trend = raw.get("trend")
        trend_label = (
            str(trend.get("label") or "").strip()
            if isinstance(trend, Mapping)
            else None
        )
        return {
            "status": "ok",
            "forecast_available": True,
            "horizon": _FORECAST_HORIZON,
            "values": values_list,
            "projections": {
                "month_1": projections.get("month_1"),
                "month_2": projections.get("month_2"),
                "month_3": projections.get("month_3"),
            },
            "tier": tier,
            "confidence_score": round(guardrail.adjusted_confidence_score, 6),
            "raw_confidence_score": round(guardrail.raw_confidence_score, 6),
            "input_points": input_points,
            "minimum_required": guardrail.minimum_required,
            "volatility_regime": volatility_regime or "unknown",
            "trend_label": trend_label,
            "warnings": warnings,
            "diagnostics": diagnostics if isinstance(diagnostics, Mapping) else {},
            "guardrail": self._guardrail_payload(guardrail),
            "raw": raw,
        }

    @staticmethod
    def _propagate_confidence(
        *,
        series_confidence: Mapping[str, Any],
        forecast: Mapping[str, Any],
        anomaly_detected: bool,
        churn_rate: float,
    ) -> tuple[float, dict[str, Any]]:
        series_score = _safe_float(series_confidence.get("confidence_score"), default=0.0)
        forecast_score = _safe_float(forecast.get("confidence_score"), default=0.0)
        forecast_available = bool(forecast.get("forecast_available"))
        forecast_status = str(forecast.get("status") or "insufficient_data")
        volatility_regime = str(forecast.get("volatility_regime") or "unknown").lower()

        if forecast_available:
            forecast_weight = _FORECAST_WEIGHT_BY_VOLATILITY.get(
                volatility_regime,
                _FORECAST_WEIGHT_BY_VOLATILITY["unknown"],
            )
            series_weight = max(0.0, 1.0 - forecast_weight)
            combined = (series_weight * series_score) + (forecast_weight * forecast_score)
        else:
            forecast_weight = 0.0
            series_weight = 0.85
            combined = 0.85 * series_score

        penalties: list[dict[str, Any]] = []
        if anomaly_detected:
            combined -= 0.08
            penalties.append({"name": "anomaly_penalty", "delta": -0.08})
        if churn_rate >= 0.10:
            combined -= 0.12
            penalties.append({"name": "high_churn_penalty", "delta": -0.12})

        if forecast_status == "insufficient_data":
            capped = min(combined, 0.45)
            if capped != combined:
                penalties.append({"name": "forecast_insufficient_cap", "delta": round(capped - combined, 6)})
            combined = capped

        if forecast_available and volatility_regime == "high":
            capped = min(combined, 0.80)
            if capped != combined:
                penalties.append({"name": "high_volatility_cap", "delta": round(capped - combined, 6)})
            combined = capped

        final_score = _clamp(combined, lower=0.0, upper=0.99)
        breakdown = {
            "series_confidence": round(series_score, 6),
            "forecast_confidence": round(forecast_score, 6),
            "forecast_status": forecast_status,
            "forecast_available": forecast_available,
            "volatility_regime": volatility_regime,
            "weights": {
                "series": round(series_weight, 6),
                "forecast": round(forecast_weight, 6),
            },
            "penalties": penalties,
            "components": list(series_confidence.get("components") or []),
            "warnings": list(series_confidence.get("warnings") or []),
        }
        return round(final_score, 6), breakdown

    def _metric_confidence(
        self,
        *,
        values: Sequence[float],
        forecast: Mapping[str, Any],
        anomaly: Mapping[str, Any],
        signal_name: str,
    ) -> tuple[float, dict[str, Any]]:
        slope_value = _slope(values)
        series_confidence = compute_confidence(
            values,
            signals={signal_name: slope_value},
        )
        score, breakdown = self._propagate_confidence(
            series_confidence=series_confidence,
            forecast=forecast,
            anomaly_detected=bool(anomaly.get("has_anomaly")),
            churn_rate=0.0,
        )
        return score, breakdown

    def analyze_business(
        self,
        *,
        entity_name: str,
        business_type: str,
        question: str,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return asyncio.run(
            self.analyze_business_async(
                entity_name=entity_name,
                business_type=business_type,
                question=question,
                context=context,
            )
        )

    async def analyze_business_async(
        self,
        *,
        entity_name: str,
        business_type: str,
        question: str,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = question
        payload = dict(context or {})
        values: list[float] = []
        revenue_series = payload.get("revenue_series")
        if isinstance(revenue_series, list):
            values = _coerce_numeric_series(revenue_series)
        else:
            if _is_numeric(payload.get("previous_revenue")):
                values.append(float(payload.get("previous_revenue")))
            if _is_numeric(payload.get("revenue")):
                values.append(float(payload.get("revenue")))

        churn_rate = _safe_float(payload.get("churn_rate"))
        conversion_rate = _safe_float(payload.get("conversion_rate"))
        total_customers = max(1.0, _safe_float(payload.get("customers"), default=1.0))
        revenue_latest = float(values[-1]) if values else _safe_float(payload.get("revenue"))
        module_results = await self._execute_insight_modules_async(
            values=values,
            payload=payload,
            business_type=business_type,
            churn_rate=churn_rate,
            conversion_rate=conversion_rate,
            total_customers=total_customers,
            revenue_latest=revenue_latest,
        )
        module_execution: dict[str, Any] = {}
        for module_name, result in module_results.items():
            item: dict[str, Any] = {
                "status": result.status,
                "execution_time_ms": result.execution_time_ms,
            }
            if result.error:
                item["error"] = result.error
            reason = str(result.payload.get("reason") or "").strip()
            if reason:
                item["reason"] = reason
            module_execution[module_name] = item

        fallback_kpis = {
            "revenue": round(revenue_latest, 6),
            "churn_rate": round(churn_rate, 6),
            "conversion_rate": round(conversion_rate, 6),
            "revenue_per_customer": round(revenue_latest / total_customers, 6),
        }
        kpi_payload = module_results["kpi_extraction"].payload
        kpis_obj = kpi_payload.get("kpis")
        kpis: dict[str, float]
        if isinstance(kpis_obj, Mapping):
            kpis = {
                key: round(_safe_float(value), 6)
                for key, value in kpis_obj.items()
            }
        else:
            kpis = fallback_kpis

        trend_payload = module_results["trend_analysis"].payload
        trend = {
            "direction": str(trend_payload.get("direction") or _trend_label(values)),
            "slope": round(_safe_float(trend_payload.get("slope"), default=_slope(values)), 6),
        }

        anomaly_payload = module_results["anomaly_detection"].payload
        anomaly = {
            "has_anomaly": bool(anomaly_payload.get("has_anomaly")),
            "outlier_count": int(anomaly_payload.get("outlier_count") or 0),
            "method": str(anomaly_payload.get("method") or "iqr"),
            "details": (
                dict(anomaly_payload.get("details"))
                if isinstance(anomaly_payload.get("details"), Mapping)
                else {}
            ),
        }

        forecast_result = module_results["forecasting"]
        if forecast_result.status in {"timeout", "error"}:
            forecast = {
                "status": "insufficient_data",
                "forecast_available": False,
                "horizon": _FORECAST_HORIZON,
                "values": [None, None, None],
                "projections": {"month_1": None, "month_2": None, "month_3": None},
                "tier": "degraded",
                "confidence_score": 0.0,
                "raw_confidence_score": 0.0,
                "input_points": len(values),
                "minimum_required": MIN_POINTS_INSUFFICIENT,
                "volatility_regime": "unknown",
                "reason": str(forecast_result.payload.get("reason") or "Forecast module unavailable."),
                "warnings": ["Forecast execution degraded due to module timeout/error."],
                "guardrail": {
                    "status": "rejected",
                    "accepted": False,
                    "reason_code": "module_unavailable",
                },
            }
        else:
            forecast = dict(forecast_result.payload)

        unit_economics_result = module_results["unit_economics"]
        if unit_economics_result.status in {"timeout", "error"}:
            unit_economics = {
                "status": "insufficient_data",
                "signals": [],
                "warnings": [
                    str(
                        unit_economics_result.payload.get("reason")
                        or "Unit economics module unavailable."
                    )
                ],
            }
        else:
            unit_economics = dict(unit_economics_result.payload)

        signals: list[str] = []
        if trend["direction"] == "down":
            signals.append("declining_revenue")
        if churn_rate >= 0.10:
            signals.append("high_churn")
        if conversion_rate <= 0.02:
            signals.append("weak_conversion")
        if anomaly["has_anomaly"]:
            signals.append("anomaly_detected")
        if forecast["status"] == "insufficient_data":
            signals.append("forecast_insufficient_data")
        if str(unit_economics.get("status") or "") != "success":
            signals.append("unit_economics_insufficient_data")
        for module_name, result in module_results.items():
            if result.status in {"timeout", "error"}:
                signals.append(f"{module_name}_degraded")
        if not signals:
            signals.append("stable_growth")

        series_confidence = compute_confidence(
            values,
            signals={
                "revenue_slope": trend["slope"],
                "conversion_rate": conversion_rate,
                "churn_inverse": -churn_rate,
            },
        )
        confidence, confidence_breakdown = self._propagate_confidence(
            series_confidence=series_confidence,
            forecast=forecast,
            anomaly_detected=bool(anomaly["has_anomaly"]),
            churn_rate=churn_rate,
        )

        insight = BusinessInsight(
            signals=signals,
            confidence_score=confidence,
            kpis=kpis,
            trend=trend,
            anomaly=anomaly,
            forecast=forecast,
            unit_economics=unit_economics,
            module_execution=module_execution,
            confidence_breakdown=confidence_breakdown,
        )
        reasoning = self._build_reasoning(insight)
        strategy = self._build_strategy(reasoning)

        data_status = (
            "ok"
            if len(values) >= MIN_POINTS_INSUFFICIENT
            else "insufficient_data"
        )

        return {
            "entity_name": entity_name,
            "business_type": business_type,
            "pipeline": {
                "data": {
                    "status": data_status,
                    "points": len(values),
                    "minimum_required_for_forecast": MIN_POINTS_INSUFFICIENT,
                    "minimum_required_for_reliable_forecast": _MIN_RELIABLE_FORECAST_POINTS,
                    "reliable_forecast_available": bool(insight.forecast.get("forecast_available")),
                },
                "insight": {
                    "signals": insight.signals,
                    "confidence_score": insight.confidence_score,
                    "kpis": insight.kpis,
                    "trend": insight.trend,
                    "anomaly": insight.anomaly,
                    "forecast": insight.forecast,
                    "unit_economics": insight.unit_economics,
                    "module_execution": insight.module_execution,
                    "confidence_breakdown": insight.confidence_breakdown,
                },
                "reasoning": reasoning,
                "strategy": strategy,
            },
            "signals_generated": insight.signals,
            "confidence_score": insight.confidence_score,
        }

    def analyze_metrics(
        self,
        *,
        entity_name: str,
        period: str,
        metrics: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        metric_results: list[dict[str, Any]] = []
        aggregate_signals: list[str] = []
        confidence_samples: list[float] = []

        for metric in metrics:
            name = str(metric.get("name") or "").strip().lower()
            raw_values = metric.get("values")
            if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes, bytearray)):
                continue
            values = _coerce_numeric_series(raw_values)
            if len(values) < 1:
                continue

            trend = _trend_label(values)
            slope_value = round(_slope(values), 6)
            anomaly = _detect_anomaly(values)
            forecast = self._safe_forecast(values)

            signals: list[str] = [f"{name}_trend_{trend}"]
            if anomaly["has_anomaly"]:
                signals.append(f"{name}_anomaly")
            if forecast["status"] == "insufficient_data":
                signals.append(f"{name}_forecast_insufficient_data")
            aggregate_signals.extend(signals)

            metric_conf, confidence_breakdown = self._metric_confidence(
                values=values,
                forecast=forecast,
                anomaly=anomaly,
                signal_name=f"{name}_slope",
            )
            confidence_samples.append(metric_conf)

            metric_results.append(
                {
                    "metric_name": name,
                    "trend": trend,
                    "slope": slope_value,
                    "anomaly": anomaly,
                    "forecast": forecast,
                    "status": (
                        "ok"
                        if forecast["status"] == "ok"
                        else "insufficient_data"
                    ),
                    "signals": signals,
                    "confidence_score": round(metric_conf, 6),
                    "confidence_breakdown": confidence_breakdown,
                }
            )

        confidence = (
            sum(confidence_samples) / float(len(confidence_samples))
            if confidence_samples
            else 0.0
        )
        if not aggregate_signals:
            aggregate_signals = ["insufficient_data"]
        return {
            "entity_name": entity_name,
            "period": period,
            "metrics_analyzed": metric_results,
            "signals_generated": aggregate_signals,
            "confidence_score": round(confidence, 6),
        }

    @staticmethod
    def _build_reasoning(insight: BusinessInsight) -> dict[str, Any]:
        risks: list[str] = []
        opportunities: list[str] = []

        if "high_churn" in insight.signals:
            risks.append("retention_risk")
        if "weak_conversion" in insight.signals:
            risks.append("funnel_efficiency_risk")
        if "declining_revenue" in insight.signals:
            risks.append("topline_decline_risk")
        if "forecast_insufficient_data" in insight.signals:
            risks.append("forecast_reliability_risk")
        if "unit_economics_insufficient_data" in insight.signals:
            risks.append("unit_economics_visibility_risk")
        if "stable_growth" in insight.signals:
            opportunities.append("scale_current_playbook")
        if "anomaly_detected" in insight.signals:
            opportunities.append("investigate_spike_or_drop")

        mode = "degraded" if risks else "normal"
        confidence = insight.confidence_score - (0.1 if mode == "degraded" else 0.0)
        confidence = max(0.0, min(0.99, confidence))

        return {
            "mode": mode,
            "risks": risks,
            "opportunities": opportunities,
            "confidence_score": round(confidence, 6),
        }

    @staticmethod
    def _build_strategy(reasoning: Mapping[str, Any]) -> dict[str, Any]:
        mode = str(reasoning.get("mode") or "normal")
        if mode == "degraded":
            return {
                "mode": "conservative",
                "actions": [
                    "stabilize_retention",
                    "tighten_spend_efficiency",
                    "run_root_cause_analysis",
                ],
            }
        return {
            "mode": "growth",
            "actions": [
                "scale_best_channels",
                "improve_conversion_experiments",
                "expand_high_confidence_segments",
            ],
        }
