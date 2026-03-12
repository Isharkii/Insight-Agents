"""
app/api/error_handling.py

Central exception handlers that enforce machine-readable API failure codes.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.failure_codes import INTERNAL_FAILURE, build_error_detail, normalize_api_failure_detail

logger = logging.getLogger(__name__)


def register_exception_handlers(application: FastAPI) -> None:
    @application.exception_handler(HTTPException)
    async def _http_exception_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        _ = request
        detail = normalize_api_failure_detail(
            status_code=exc.status_code,
            detail=exc.detail,
        )
        context = detail.get("context")
        if not isinstance(context, dict):
            context = {}
        request_id = str(getattr(request.state, "request_id", "") or "").strip()
        if request_id:
            context["request_id"] = request_id
        if context:
            detail["context"] = context
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": detail},
            headers=exc.headers,
        )

    @application.exception_handler(RequestValidationError)
    async def _request_validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        _ = request
        detail = normalize_api_failure_detail(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.errors(),
        )
        context = detail.get("context")
        if not isinstance(context, dict):
            context = {}
        request_id = str(getattr(request.state, "request_id", "") or "").strip()
        if request_id:
            context["request_id"] = request_id
        if context:
            detail["context"] = context
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": detail},
        )

    @application.exception_handler(Exception)
    async def _unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.exception(
            "Unhandled API exception path=%s method=%s error=%s",
            request.url.path,
            request.method,
            exc,
        )
        context = {"exception_type": type(exc).__name__}
        request_id = str(getattr(request.state, "request_id", "") or "").strip()
        if request_id:
            context["request_id"] = request_id
        detail = build_error_detail(
            code=INTERNAL_FAILURE,
            message="Internal server error.",
            context=context,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": detail},
        )
