from __future__ import annotations

import logging
import re
import secrets
from typing import Any

from starlette.requests import Request

from app.failure_codes import AUTHENTICATION_FAILED, AUTHORIZATION_FAILED, INTERNAL_FAILURE
from app.security.errors import SecurityError
from app.security.jwt_utils import decode_and_verify_hs256_jwt
from app.security.models import SecurityContext
from app.security.settings import SecuritySettings

logger = logging.getLogger(__name__)

_TENANT_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,63}$")


def _sanitize_tenant_id(value: str | None) -> str:
    tenant_id = str(value or "").strip()
    if not tenant_id or not _TENANT_RE.match(tenant_id):
        raise SecurityError(
            status_code=400,
            code=AUTHORIZATION_FAILED,
            message="Invalid tenant identifier.",
        )
    return tenant_id


def _extract_scopes(claims: dict[str, Any]) -> frozenset[str]:
    raw_scope = claims.get("scope")
    if isinstance(raw_scope, str):
        scopes = {part for part in raw_scope.strip().split(" ") if part}
        return frozenset(scopes)
    raw_scopes = claims.get("scopes")
    if isinstance(raw_scopes, list):
        scopes = {str(part).strip() for part in raw_scopes if str(part).strip()}
        return frozenset(scopes)
    return frozenset()


def _extract_allowed_entities(claims: dict[str, Any]) -> frozenset[str]:
    raw = claims.get("entities")
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(
        str(item).strip().lower()
        for item in raw
        if str(item).strip()
    )


class RequestAuthenticator:
    def __init__(self, settings: SecuritySettings) -> None:
        self._settings = settings

    def authenticate(self, request: Request, *, request_id: str) -> SecurityContext:
        if not self._settings.enabled:
            tenant = request.headers.get(self._settings.tenant_header_name) or "public"
            tenant_id = _sanitize_tenant_id(tenant)
            return SecurityContext(
                request_id=request_id,
                tenant_id=tenant_id,
                subject="anonymous",
                auth_type="disabled",
                scopes=frozenset({"*"}),
            )

        authz = str(request.headers.get(self._settings.authorization_header_name) or "").strip()
        api_key = str(request.headers.get(self._settings.api_key_header_name) or "").strip()

        if authz.lower().startswith("bearer "):
            token = authz[7:].strip()
            if not token:
                raise SecurityError(
                    status_code=401,
                    code=AUTHENTICATION_FAILED,
                    message="Missing bearer token.",
                )
            return self._authenticate_jwt(token=token, request=request, request_id=request_id)

        if api_key:
            return self._authenticate_api_key(api_key=api_key, request=request, request_id=request_id)

        raise SecurityError(
            status_code=401,
            code=AUTHENTICATION_FAILED,
            message="Authentication required. Provide X-API-Key or Bearer token.",
        )

    def _authenticate_jwt(
        self,
        *,
        token: str,
        request: Request,
        request_id: str,
    ) -> SecurityContext:
        if not self._settings.jwt_secret_key:
            raise SecurityError(
                status_code=500,
                code=INTERNAL_FAILURE,
                message="JWT authentication is not configured.",
            )

        claims = decode_and_verify_hs256_jwt(
            token,
            secret_key=self._settings.jwt_secret_key,
            issuer=self._settings.jwt_issuer,
            audience=self._settings.jwt_audience,
        )

        tenant_claim = (
            claims.get("tenant_id")
            or claims.get("tid")
            or claims.get("tenant")
        )
        tenant_id = _sanitize_tenant_id(str(tenant_claim or ""))

        header_tenant = request.headers.get(self._settings.tenant_header_name)
        if header_tenant and _sanitize_tenant_id(header_tenant) != tenant_id:
            raise SecurityError(
                status_code=403,
                code=AUTHORIZATION_FAILED,
                message="Tenant header does not match JWT tenant.",
            )

        subject = str(claims.get("sub") or "").strip() or f"tenant:{tenant_id}"
        scopes = _extract_scopes(claims)
        allowed_entities = _extract_allowed_entities(claims)

        return SecurityContext(
            request_id=request_id,
            tenant_id=tenant_id,
            subject=subject,
            auth_type="jwt",
            scopes=scopes,
            claims=claims,
            allowed_entities=allowed_entities,
        )

    def _authenticate_api_key(
        self,
        *,
        api_key: str,
        request: Request,
        request_id: str,
    ) -> SecurityContext:
        matched_tenant: str | None = None
        allowed_entities = frozenset()

        for key_value, tenant_id in self._settings.key_to_tenant.items():
            if secrets.compare_digest(api_key, key_value):
                matched_tenant = tenant_id
                allowed_entities = self._settings.tenant_entities.get(tenant_id, frozenset())
                break

        if matched_tenant is None and self._settings.legacy_api_key:
            if secrets.compare_digest(api_key, self._settings.legacy_api_key):
                header_tenant = request.headers.get(self._settings.tenant_header_name)
                matched_tenant = _sanitize_tenant_id(header_tenant)

        if matched_tenant is None:
            logger.warning("API key auth failed path=%s request_id=%s", request.url.path, request_id)
            raise SecurityError(
                status_code=401,
                code=AUTHENTICATION_FAILED,
                message="Invalid API key.",
            )

        return SecurityContext(
            request_id=request_id,
            tenant_id=matched_tenant,
            subject=f"api_key:{matched_tenant}",
            auth_type="api_key",
            scopes=frozenset({"*"}),
            claims={},
            allowed_entities=allowed_entities,
        )
