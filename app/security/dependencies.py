from __future__ import annotations

from uuid import uuid4

from fastapi import HTTPException, Request, status

from app.failure_codes import AUTHORIZATION_FAILED, INTERNAL_FAILURE, build_error_detail
from app.security.models import SecurityContext

TENANT_ENTITY_SEPARATOR = "::"


def require_security_context(request: Request) -> SecurityContext:
    """
    FastAPI dependency to fetch request-scoped security context.

    Falls back to a minimal context in tests that call routers directly
    without middleware.
    """
    context = getattr(request.state, "security_context", None)
    if isinstance(context, SecurityContext):
        return context

    request_id = str(getattr(request.state, "request_id", "") or "").strip()
    if not request_id:
        request_id = str(request.headers.get("X-Request-ID") or "").strip() or str(uuid4())
    tenant_id = str(request.headers.get("X-Tenant-ID") or "").strip() or "legacy"
    return SecurityContext(
        request_id=request_id,
        tenant_id=tenant_id,
        subject="legacy",
        auth_type="legacy",
        scopes=frozenset({"*"}),
        claims={},
    )


def assert_entity_allowed_for_tenant(
    *,
    entity_name: str,
    security: SecurityContext,
) -> None:
    """
    Enforce tenant-level entity access rules when claim-level entity allowlists exist.
    """
    resolved = str(entity_name or "").strip()
    if not resolved:
        return

    if TENANT_ENTITY_SEPARATOR in resolved:
        tenant_prefix, _, _ = resolved.partition(TENANT_ENTITY_SEPARATOR)
        if tenant_prefix and tenant_prefix != security.tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=build_error_detail(
                    code=AUTHORIZATION_FAILED,
                    message="Entity does not belong to the authenticated tenant.",
                    context={
                        "tenant_id": security.tenant_id,
                        "request_id": security.request_id,
                    },
                ),
            )

    normalized = resolved.lower()
    _, has_sep, local_entity = resolved.partition(TENANT_ENTITY_SEPARATOR)
    local_normalized = local_entity.lower() if has_sep else normalized
    if (
        security.allowed_entities
        and normalized not in security.allowed_entities
        and local_normalized not in security.allowed_entities
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=build_error_detail(
                code=AUTHORIZATION_FAILED,
                message="Entity access denied for tenant policy.",
                context={
                    "tenant_id": security.tenant_id,
                    "request_id": security.request_id,
                },
            ),
        )


def require_scope(
    *,
    security: SecurityContext,
    scope: str,
) -> None:
    if security.has_scope(scope):
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=build_error_detail(
            code=AUTHORIZATION_FAILED,
            message="Insufficient scope for this operation.",
            context={
                "required_scope": scope,
                "tenant_id": security.tenant_id,
                "request_id": security.request_id,
            },
        ),
    )


def request_id_from(request: Request) -> str:
    value = str(getattr(request.state, "request_id", "") or "").strip()
    if value:
        return value
    header = str(request.headers.get("X-Request-ID") or "").strip()
    return header or str(uuid4())


def tenant_from_request(request: Request) -> str:
    context = getattr(request.state, "security_context", None)
    if isinstance(context, SecurityContext):
        return context.tenant_id
    tenant_id = str(request.headers.get("X-Tenant-ID") or "").strip()
    if tenant_id:
        return tenant_id
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=build_error_detail(
            code=INTERNAL_FAILURE,
            message="Tenant context unavailable.",
        ),
    )
