"""
app/api/routers/decision_engine_router.py

Synchronous deployment layer for the decision intelligence engine.

Endpoints:
    POST /analyze_business
    POST /analyze_metrics
    GET  /system_health
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Any, Literal, Mapping

from fastapi import APIRouter, Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.failure_codes import AUTHENTICATION_FAILED, INTERNAL_FAILURE, build_error_detail
from app.security.dependencies import (
    assert_entity_allowed_for_tenant,
    request_id_from,
    require_security_context,
)
from app.security.models import SecurityContext
from app.services.decision_engine_service import DecisionEngineService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["decision-engine"])
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class ApiError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    context: dict[str, Any] | None = None


class ApiResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    success: bool
    data: dict[str, Any] | None = None
    error: ApiError | None = None


class AnalyzeBusinessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_name: str = Field(min_length=1, max_length=128)
    business_type: Literal["saas", "ecommerce", "agency", "general_timeseries"]
    question: str = Field(min_length=3, max_length=500)
    context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("context")
    @classmethod
    def _validate_context_size(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 100:
            raise ValueError("context may contain at most 100 keys")
        return value


class MetricSeries(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    values: list[float] = Field(min_length=1, max_length=500)


class AnalyzeMetricsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_name: str = Field(min_length=1, max_length=128)
    period: Literal["daily", "weekly", "monthly"] = "monthly"
    metrics: list[MetricSeries] = Field(min_length=1, max_length=100)


def _request_id_from(request: Request) -> str:
    return request_id_from(request)


def _api_key_required(
    request: Request,
    api_key: str | None = Security(_api_key_header),
) -> None:
    if hasattr(request.state, "security_context"):
        # Global security middleware already validated this request.
        return
    # Backward-compatible env options:
    #   DECISION_ENGINE_API_KEY (preferred)
    #   DECISION_API_KEY (legacy)
    configured = (
        os.getenv("DECISION_ENGINE_API_KEY", "").strip()
        or os.getenv("DECISION_API_KEY", "").strip()
    )
    if not configured:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(
                code=INTERNAL_FAILURE,
                message="Server API key is not configured.",
            ),
        )
    if not api_key or not secrets.compare_digest(api_key, configured):
        logger.warning(
            "Decision engine auth failed request_id=%s path=%s",
            _request_id_from(request),
            request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=build_error_detail(
                code=AUTHENTICATION_FAILED,
                message="Invalid API key.",
            ),
        )


def _service() -> DecisionEngineService:
    return DecisionEngineService()


def _ok(request_id: str, data: Mapping[str, Any]) -> ApiResponse:
    return ApiResponse(
        request_id=request_id,
        success=True,
        data=dict(data),
        error=None,
    )


@router.post(
    "/analyze_business",
    response_model=ApiResponse,
    dependencies=[Depends(_api_key_required)],
)
def analyze_business(
    body: AnalyzeBusinessRequest,
    request: Request,
    service: DecisionEngineService = Depends(_service),
    security: SecurityContext = Depends(require_security_context),
) -> ApiResponse:
    request_id = _request_id_from(request)
    assert_entity_allowed_for_tenant(entity_name=body.entity_name, security=security)
    try:
        result = service.analyze_business(
            entity_name=body.entity_name,
            business_type=body.business_type,
            question=body.question,
            context=body.context,
        )
        result = dict(result)
        result.setdefault("tenant_id", security.tenant_id)
        logger.info(
            "analyze_business success request_id=%s tenant=%s entity=%s signals=%d",
            request_id,
            security.tenant_id,
            body.entity_name,
            len(result.get("signals_generated", [])),
        )
        return _ok(request_id, result)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "analyze_business failed request_id=%s entity=%s error=%s",
            request_id,
            body.entity_name,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(
                code=INTERNAL_FAILURE,
                message="Business analysis failed.",
                context={"request_id": request_id, "exception_type": type(exc).__name__},
            ),
        ) from exc


@router.post(
    "/analyze_metrics",
    response_model=ApiResponse,
    dependencies=[Depends(_api_key_required)],
)
def analyze_metrics(
    body: AnalyzeMetricsRequest,
    request: Request,
    service: DecisionEngineService = Depends(_service),
    security: SecurityContext = Depends(require_security_context),
) -> ApiResponse:
    request_id = _request_id_from(request)
    assert_entity_allowed_for_tenant(entity_name=body.entity_name, security=security)
    try:
        metrics_payload: list[dict[str, Any]] = [
            {"name": item.name, "values": item.values}
            for item in body.metrics
        ]
        result = service.analyze_metrics(
            entity_name=body.entity_name,
            period=body.period,
            metrics=metrics_payload,
        )
        result = dict(result)
        result.setdefault("tenant_id", security.tenant_id)
        logger.info(
            "analyze_metrics success request_id=%s tenant=%s entity=%s metrics=%d",
            request_id,
            security.tenant_id,
            body.entity_name,
            len(body.metrics),
        )
        return _ok(request_id, result)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "analyze_metrics failed request_id=%s entity=%s error=%s",
            request_id,
            body.entity_name,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(
                code=INTERNAL_FAILURE,
                message="Metric analysis failed.",
                context={"request_id": request_id, "exception_type": type(exc).__name__},
            ),
        ) from exc


@router.get("/system_health", response_model=ApiResponse)
def system_health(request: Request) -> ApiResponse:
    request_id = _request_id_from(request)
    # Keep synchronous and deterministic with zero external dependencies.
    data = {
        "status": "ok",
        "service": "decision_engine",
        "checks": {
            "api": "ok",
            "decision_engine_service": "ok",
            "auth_configured": bool(
                os.getenv("DECISION_ENGINE_API_KEY", "").strip()
                or os.getenv("DECISION_API_KEY", "").strip()
            ),
        },
    }
    return _ok(request_id, data)
