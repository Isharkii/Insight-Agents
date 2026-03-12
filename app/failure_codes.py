"""Shared failure code constants and helpers for API error handling."""

from __future__ import annotations

from typing import Any, Literal, Mapping

FailureCode = Literal[
    "ingestion_validation",
    "schema_conflict",
    "insufficient_history",
    "external_unavailable",
    "authentication_failed",
    "authorization_failed",
    "rate_limited",
    "internal_failure",
]

INGESTION_VALIDATION: FailureCode = "ingestion_validation"
SCHEMA_CONFLICT: FailureCode = "schema_conflict"
INSUFFICIENT_HISTORY: FailureCode = "insufficient_history"
EXTERNAL_UNAVAILABLE: FailureCode = "external_unavailable"
AUTHENTICATION_FAILED: FailureCode = "authentication_failed"
AUTHORIZATION_FAILED: FailureCode = "authorization_failed"
RATE_LIMITED: FailureCode = "rate_limited"
INTERNAL_FAILURE: FailureCode = "internal_failure"

ALL_FAILURE_CODES: frozenset[str] = frozenset(
    {
        INGESTION_VALIDATION,
        SCHEMA_CONFLICT,
        INSUFFICIENT_HISTORY,
        EXTERNAL_UNAVAILABLE,
        AUTHENTICATION_FAILED,
        AUTHORIZATION_FAILED,
        RATE_LIMITED,
        INTERNAL_FAILURE,
    }
)

# Legacy constants still used by ingestion orchestration paths.
CRITICAL_FAILURES = [
    "empty_kpi",
    "empty_forecast",
    "empty_risk",
]

OPTIONAL_FAILURES = [
    "missing_segmentation",
    "missing_scraping",
]


def normalize_failure_code(value: Any, *, default: FailureCode = INTERNAL_FAILURE) -> FailureCode:
    text = str(value or "").strip().lower()
    if text in ALL_FAILURE_CODES:
        return text  # type: ignore[return-value]
    return default


def _flatten_text(detail: Any) -> str:
    if isinstance(detail, Mapping):
        parts: list[str] = []
        for key in ("message", "error", "detail", "error_type", "code"):
            if key in detail and detail.get(key) is not None:
                parts.append(str(detail.get(key)))
        if not parts:
            for value in detail.values():
                if value is None:
                    continue
                if isinstance(value, (str, int, float, bool)):
                    parts.append(str(value))
        return " ".join(parts).strip().lower()
    if isinstance(detail, list):
        return " ".join(str(item) for item in detail).strip().lower()
    return str(detail or "").strip().lower()


def infer_failure_code(*, status_code: int, detail: Any = None) -> FailureCode:
    if isinstance(detail, Mapping):
        explicit = str(detail.get("code") or "").strip().lower()
        if explicit in ALL_FAILURE_CODES:
            return explicit  # type: ignore[return-value]

    text = _flatten_text(detail)

    if any(token in text for token in ("insufficient", "no kpi records", "insufficient_data")):
        return INSUFFICIENT_HISTORY
    if any(
        token in text
        for token in (
            "external",
            "connector",
            "newsapi",
            "world bank",
            "google trends",
            "timeout",
            "competitor",
            "scraping",
            "unavailable",
        )
    ):
        return EXTERNAL_UNAVAILABLE
    if any(
        token in text
        for token in (
            "ingestion",
            "csv",
            "header",
            "validation",
            "row",
            "canonical_validation_failed",
        )
    ):
        return INGESTION_VALIDATION
    if any(
        token in text
        for token in (
            "api key",
            "api_key",
            "bearer",
            "jwt",
            "auth",
            "token",
            "unauthorized",
            "authentication",
        )
    ):
        return AUTHENTICATION_FAILED
    if any(
        token in text
        for token in (
            "forbidden",
            "scope",
            "tenant",
            "permission",
            "authorization",
            "not allowed",
            "access denied",
        )
    ):
        return AUTHORIZATION_FAILED
    if any(
        token in text
        for token in (
            "rate limit",
            "too many requests",
            "throttle",
            "retry-after",
        )
    ):
        return RATE_LIMITED
    if any(
        token in text
        for token in (
            "schema",
            "mapping",
            "unsupported",
            "invalid",
            "not found",
            "entity",
            "format",
            "dataset",
            "client",
            "conflict",
        )
    ):
        return SCHEMA_CONFLICT

    if status_code >= 500:
        return INTERNAL_FAILURE
    if status_code == 429:
        return RATE_LIMITED
    if status_code == 401:
        return AUTHENTICATION_FAILED
    if status_code == 403:
        return AUTHORIZATION_FAILED
    if status_code in {400, 404, 409, 422}:
        return SCHEMA_CONFLICT
    return INTERNAL_FAILURE


def build_error_detail(
    *,
    code: FailureCode,
    message: str,
    context: Mapping[str, Any] | None = None,
    error_type: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code": normalize_failure_code(code),
        "message": str(message or "").strip() or "Request failed.",
    }
    if error_type:
        payload["error_type"] = str(error_type)
    if context:
        payload["context"] = dict(context)
    return payload


def normalize_api_failure_detail(*, status_code: int, detail: Any) -> dict[str, Any]:
    if isinstance(detail, Mapping):
        code = infer_failure_code(status_code=status_code, detail=detail)
        message = str(
            detail.get("message")
            or detail.get("detail")
            or detail.get("error")
            or "Request failed."
        ).strip() or "Request failed."
        normalized: dict[str, Any] = dict(detail)
        normalized["code"] = code
        normalized["message"] = message
        return normalized

    if isinstance(detail, list):
        code = infer_failure_code(status_code=status_code, detail=detail)
        return build_error_detail(
            code=code,
            message="Request validation failed.",
            context={"errors": detail},
        )

    code = infer_failure_code(status_code=status_code, detail=detail)
    return build_error_detail(
        code=code,
        message=str(detail or "Request failed."),
    )
