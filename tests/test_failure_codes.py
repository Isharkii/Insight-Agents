from __future__ import annotations

from app.failure_codes import (
    EXTERNAL_UNAVAILABLE,
    INGESTION_VALIDATION,
    INTERNAL_FAILURE,
    SCHEMA_CONFLICT,
    infer_failure_code,
    normalize_api_failure_detail,
)


def test_infer_failure_code_from_keywords() -> None:
    assert infer_failure_code(status_code=400, detail="CSV validation failed.") == INGESTION_VALIDATION
    assert infer_failure_code(status_code=400, detail="Schema mapping conflict.") == SCHEMA_CONFLICT
    assert infer_failure_code(status_code=503, detail="External connector unavailable.") == EXTERNAL_UNAVAILABLE
    assert infer_failure_code(status_code=500, detail="Unhandled exception.") == INTERNAL_FAILURE


def test_normalize_api_failure_detail_preserves_context_and_code() -> None:
    payload = normalize_api_failure_detail(
        status_code=400,
        detail={
            "message": "Invalid dataset.",
            "context": {"dataset": "unknown"},
        },
    )
    assert payload["code"] == SCHEMA_CONFLICT
    assert payload["message"] == "Invalid dataset."
    assert payload["context"]["dataset"] == "unknown"


def test_normalize_api_failure_detail_from_list_errors() -> None:
    payload = normalize_api_failure_detail(
        status_code=422,
        detail=[{"loc": ["body", "field"], "msg": "field required"}],
    )
    assert payload["code"] == SCHEMA_CONFLICT
    assert payload["message"] == "Request validation failed."
    assert isinstance(payload["context"]["errors"], list)
