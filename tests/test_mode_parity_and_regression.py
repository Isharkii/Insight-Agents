from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import app.config as app_config
from app.mappers.schema_mapper import SchemaMapper
from app.services.kpi_orchestrator import KPIOrchestrator, _AggregatedInputs, _build_payload
from llm_synthesis.schema import InsightOutput


def _run_with_mode(
    monkeypatch,
    mode: str,
    fn: Callable[[], Any],
) -> Any:
    monkeypatch.setenv("APP_MODE", mode)
    app_config.get_app_settings.cache_clear()
    settings = app_config.get_app_settings()
    assert settings.mode == mode
    return fn()


def _schema_mapping_snapshot() -> dict[str, Any]:
    headers = ("entity_name", "category", "metric_name", "metric_value", "timestamp")
    mapper = SchemaMapper(category_hint="saas")
    resolved = mapper.resolve_mapping(headers=headers, category_hint="saas")
    return {
        "mapping": resolved.canonical_to_source,
        "strategies": resolved.match_strategies,
        "sources": list(resolved.source_headers),
    }


def _kpi_payload_snapshot() -> dict[str, Any]:
    agg = _AggregatedInputs(
        subscription_revenues=[120.0, 180.0],
        active_customers=100,
        lost_customers=5,
        arpu=3.0,
        current_revenue=300.0,
        previous_revenue=240.0,
    )
    metrics, validity = KPIOrchestrator()._compute(
        business_type="saas",
        agg_inputs=agg,
        extra_inputs={},
    )
    return _build_payload(metrics, validity)


def _strict_synthesis_snapshot() -> dict[str, Any]:
    payload = {
        "insight": "Stable insight",
        "evidence": "Stable evidence",
        "impact": "Stable impact",
        "recommended_action": "Stable action",
        "priority": "medium",
        "confidence_score": 0.9,
        "pipeline_status": "success",
        "diagnostics": {
            "warnings": [],
            "confidence_score": 1.0,
            "missing_signal": [],
            "confidence_adjustments": [],
        },
    }
    validated = InsightOutput.model_validate(payload)
    return {
        "fields": set(InsightOutput.model_fields.keys()),
        "schema": InsightOutput.model_json_schema(),
        "payload": validated.model_dump(),
    }


def test_identical_dataset_produces_identical_outputs() -> None:
    dataset = [
        {
            "entity_name": "Acme",
            "category": "saas",
            "metric_name": "mrr",
            "metric_value": "120.0",
            "timestamp": "2026-01-01T00:00:00+00:00",
        },
        {
            "entity_name": "Acme",
            "category": "saas",
            "metric_name": "mrr",
            "metric_value": "180.0",
            "timestamp": "2026-02-01T00:00:00+00:00",
        },
    ]

    mapper = SchemaMapper(category_hint="saas")
    mapping = mapper.resolve_mapping(headers=tuple(dataset[0].keys()), category_hint="saas")

    mapped_once = [mapper.map_row(raw_row=row, mapping=mapping) for row in dataset]
    mapped_twice = [mapper.map_row(raw_row=row, mapping=mapping) for row in copy.deepcopy(dataset)]
    assert mapped_once == mapped_twice

    payload_once = _kpi_payload_snapshot()
    payload_twice = _kpi_payload_snapshot()
    assert payload_once == payload_twice
    assert json.dumps(payload_once, sort_keys=True) == json.dumps(payload_twice, sort_keys=True)


def test_local_cloud_parity_for_schema_mapping(monkeypatch) -> None:
    local = _run_with_mode(monkeypatch, "local", _schema_mapping_snapshot)
    cloud = _run_with_mode(monkeypatch, "cloud", _schema_mapping_snapshot)
    assert local == cloud


def test_local_cloud_parity_for_kpi_math(monkeypatch) -> None:
    local = _run_with_mode(monkeypatch, "local", _kpi_payload_snapshot)
    cloud = _run_with_mode(monkeypatch, "cloud", _kpi_payload_snapshot)
    assert local == cloud


def test_local_cloud_parity_for_strict_json_synthesis_schema(monkeypatch) -> None:
    local = _run_with_mode(monkeypatch, "local", _strict_synthesis_snapshot)
    cloud = _run_with_mode(monkeypatch, "cloud", _strict_synthesis_snapshot)
    assert local == cloud
