from __future__ import annotations

import logging
from uuid import uuid4

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.failure_codes import RATE_LIMITED, build_error_detail
from app.security.auth import RequestAuthenticator
from app.security.errors import SecurityError
from app.security.models import SecurityContext
from app.security.rate_limit import SlidingWindowRateLimiter
from app.security.settings import SecuritySettings, get_security_settings

logger = logging.getLogger(__name__)


def resolve_request_id(request: Request, *, header_name: str = "X-Request-ID") -> str:
    incoming = str(request.headers.get(header_name) or "").strip()
    if incoming:
        return incoming[:128]
    return str(uuid4())


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Cross-cutting API security middleware:
      - request ID propagation
      - auth (API key or JWT)
      - tenant context
      - rate limiting
      - structured security errors
    """

    def __init__(self, app, settings: SecuritySettings | None = None) -> None:
        super().__init__(app)
        self._settings = settings or get_security_settings()
        self._authenticator = RequestAuthenticator(self._settings)
        self._limiter = SlidingWindowRateLimiter(
            max_requests=self._settings.rate_limit_max_requests,
            window_seconds=self._settings.rate_limit_window_seconds,
        )

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = resolve_request_id(
            request,
            header_name=self._settings.request_id_header_name,
        )
        request.state.request_id = request_id
        path = request.url.path

        if self._is_public_request(path=path, method=request.method):
            anonymous_context = SecurityContext(
                request_id=request_id,
                tenant_id="public",
                subject="anonymous",
                auth_type="public",
                scopes=frozenset(),
            )
            request.state.security_context = anonymous_context
            response = await call_next(request)
            response.headers[self._settings.request_id_header_name] = request_id
            return response

        context: SecurityContext | None = None
        try:
            context = self._authenticator.authenticate(request, request_id=request_id)
            request.state.security_context = context
            request.state.tenant_id = context.tenant_id

            if self._settings.rate_limit_enabled:
                key = f"{context.tenant_id}:{path}"
                allowed, retry_after = self._limiter.allow(key)
                if not allowed:
                    raise SecurityError(
                        status_code=429,
                        code=RATE_LIMITED,
                        message="Rate limit exceeded.",
                        context={
                            "tenant_id": context.tenant_id,
                            "path": path,
                            "window_seconds": self._settings.rate_limit_window_seconds,
                            "limit": self._settings.rate_limit_max_requests,
                        },
                        retry_after_seconds=retry_after,
                    )

            response = await call_next(request)
        except SecurityError as exc:
            error_context = dict(exc.context)
            error_context["request_id"] = request_id
            error_context["path"] = path
            detail = build_error_detail(
                code=exc.code,  # type: ignore[arg-type]
                message=exc.message,
                context=error_context,
            )
            response = JSONResponse(
                status_code=exc.status_code,
                content={"detail": detail},
            )
            if exc.retry_after_seconds is not None:
                response.headers["Retry-After"] = str(exc.retry_after_seconds)
            logger.warning(
                "Security middleware rejected request path=%s request_id=%s status=%s code=%s",
                path,
                request_id,
                exc.status_code,
                exc.code,
            )

        response.headers[self._settings.request_id_header_name] = request_id
        if context is not None:
            response.headers[self._settings.tenant_header_name] = context.tenant_id
        return response

    def _is_public_request(self, *, path: str, method: str) -> bool:
        if method.upper() == "OPTIONS":
            return True
        for prefix in self._settings.public_path_prefixes:
            if path.startswith(prefix):
                return True
        return False
