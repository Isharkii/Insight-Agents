"""
agent/nodes/unit_economics_node.py

Compute deterministic unit economics metrics and health signals from KPI data.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agent.helpers.confidence_model import compute_standard_confidence
from agent.helpers.kpi_extraction import (
    dataset_confidence_from_state,
    metric_series_from_kpi_payload,
    records_from_kpi_payload,
    resolve_kpi_payload,
)
from agent.nodes.node_result import failed, insufficient_data, skipped, success
from agent.state import AgentState
from app.services.unit_economics import analyze_unit_economics


def _synthetic_records_from_metric_series(metric_series: dict[str, list[float]]) -> list[dict]:
    """Build minimal KPI-like records from metric series for legacy analyzers."""
    if not metric_series:
        return []
    max_len = max((len(values) for values in metric_series.values()), default=0)
    if max_len <= 0:
        return []

    start = datetime(2000, 1, 1, tzinfo=timezone.utc)
    records: list[dict] = []
    for idx in range(max_len):
        computed: dict[str, dict[str, float]] = {}
        for metric_name, values in metric_series.items():
            if idx >= len(values):
                continue
            computed[str(metric_name)] = {"value": float(values[idx])}
        if not computed:
            continue
        ts = (start + timedelta(days=idx)).isoformat()
        records.append(
            {
                "period_start": ts,
                "period_end": ts,
                "created_at": ts,
                "computed_kpis": computed,
            }
        )
    return records


def unit_economics_node(state: AgentState) -> AgentState:
    """Run unit economics analysis over the resolved KPI payload records."""
    try:
        kpi_payload = resolve_kpi_payload(state)
        records = records_from_kpi_payload(kpi_payload)
        metric_series = metric_series_from_kpi_payload(kpi_payload)
        if not records and metric_series:
            records = _synthetic_records_from_metric_series(metric_series)
        if not records:
            return {
                "unit_economics_data": skipped("kpi_unavailable", {"records": 0}),
            }

        business_type = str(state.get("business_type") or "").strip().lower()
        result = analyze_unit_economics(records, business_type=business_type)

        warnings = [str(item) for item in result.get("warnings", []) if str(item).strip()]
        dataset_confidence = dataset_confidence_from_state(state)
        if dataset_confidence < 1.0:
            warnings.append(
                f"Dataset confidence reduced unit economics reliability ({dataset_confidence:.2f})."
            )
        metric_series = result.get("metric_series")
        if not isinstance(metric_series, dict):
            metric_series = {}
        values: list[float] = []
        for preferred_metric in ("revenue", "growth_rate", "ltv", "cac"):
            candidate = metric_series.get(preferred_metric)
            if isinstance(candidate, list) and candidate:
                values = [float(v) for v in candidate if isinstance(v, (int, float))]
                if values:
                    break
        if not values:
            metrics = result.get("metrics")
            if isinstance(metrics, dict):
                values = [
                    float(v)
                    for v in metrics.values()
                    if isinstance(v, (int, float))
                ]

        trend_signals: dict[str, float | None] = {}
        trends = result.get("trends")
        if isinstance(trends, dict):
            for name, value in trends.items():
                if isinstance(value, (int, float)):
                    trend_signals[f"trend_{name}"] = float(value)

        severity_to_score = {"info": 0.3, "warning": -0.4, "critical": -1.0}
        signal_payload = result.get("signals")
        if isinstance(signal_payload, list):
            for item in signal_payload:
                if not isinstance(item, dict):
                    continue
                signal_name = str(item.get("signal") or "").strip()
                if not signal_name:
                    continue
                severity = str(item.get("severity") or "warning").strip().lower()
                trend_signals[f"ue_{signal_name}"] = severity_to_score.get(severity, -0.2)

        confidence_model = compute_standard_confidence(
            values=values,
            signals=trend_signals,
            dataset_confidence=dataset_confidence,
            upstream_confidences=[float(result.get("confidence") or 0.0)],
            status=str(result.get("status") or "success"),
            base_warnings=warnings,
        )
        result_payload = {
            **result,
            "confidence_breakdown": confidence_model,
        }

        status = str(result.get("status") or "").strip().lower()
        if status == "insufficient_data":
            reason = ""
            signals = result.get("signals")
            if isinstance(signals, list) and signals:
                first = signals[0]
                if isinstance(first, dict):
                    reason = str(first.get("description") or "")
            if not reason:
                reason = "Insufficient data for unit economics analysis."
            envelope = insufficient_data(
                reason,
                payload=result_payload,
                warnings=confidence_model["warnings"],
                confidence_score=float(confidence_model["confidence_score"]),
            )
            return {"unit_economics_data": envelope}

        return {
            "unit_economics_data": success(
                result_payload,
                warnings=confidence_model["warnings"],
                confidence_score=float(confidence_model["confidence_score"]),
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "unit_economics_data": failed(str(exc), {"stage": "unit_economics"}),
        }
