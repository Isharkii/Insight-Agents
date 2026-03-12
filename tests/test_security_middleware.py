from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Any

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.security.middleware import SecurityMiddleware
from app.security.settings import get_security_settings


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _build_hs256_jwt(payload: dict[str, Any], *, secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    signature_b64 = _b64url(signature)
    return f"{header_b64}.{payload_b64}.{signature_b64}"


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(SecurityMiddleware)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/secure")
    async def secure(request: Request) -> dict[str, str]:
        context = request.state.security_context
        return {
            "tenant_id": context.tenant_id,
            "subject": context.subject,
            "auth_type": context.auth_type,
            "request_id": request.state.request_id,
        }

    return app


def _reset_security_cache() -> None:
    get_security_settings.cache_clear()


def test_api_key_auth_sets_tenant_and_request_id(monkeypatch) -> None:
    monkeypatch.setenv("API_SECURITY_ENABLED", "true")
    monkeypatch.setenv("API_RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("API_SECURITY_PUBLIC_PATHS", "/health")
    monkeypatch.setenv("API_KEYS_JSON", json.dumps({"tenant_a": "api-key-a"}))
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    _reset_security_cache()

    client = TestClient(_build_app())
    response = client.get(
        "/secure",
        headers={
            "X-API-Key": "api-key-a",
            "X-Request-ID": "req-security-1",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == "tenant_a"
    assert body["auth_type"] == "api_key"
    assert body["request_id"] == "req-security-1"
    assert response.headers["X-Request-ID"] == "req-security-1"
    assert response.headers["X-Tenant-ID"] == "tenant_a"


def test_missing_auth_returns_structured_401(monkeypatch) -> None:
    monkeypatch.setenv("API_SECURITY_ENABLED", "true")
    monkeypatch.setenv("API_RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("API_SECURITY_PUBLIC_PATHS", "/health")
    monkeypatch.setenv("API_KEYS_JSON", json.dumps({"tenant_a": "api-key-a"}))
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    _reset_security_cache()

    client = TestClient(_build_app())
    response = client.get("/secure")
    assert response.status_code == 401
    detail = response.json()["detail"]
    assert detail["code"] == "authentication_failed"
    assert "request_id" in detail.get("context", {})


def test_rate_limit_returns_429_with_retry_after(monkeypatch) -> None:
    monkeypatch.setenv("API_SECURITY_ENABLED", "true")
    monkeypatch.setenv("API_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("API_RATE_LIMIT_MAX_REQUESTS", "1")
    monkeypatch.setenv("API_RATE_LIMIT_WINDOW_SECONDS", "60")
    monkeypatch.setenv("API_SECURITY_PUBLIC_PATHS", "/health")
    monkeypatch.setenv("API_KEYS_JSON", json.dumps({"tenant_a": "api-key-a"}))
    _reset_security_cache()

    client = TestClient(_build_app())
    first = client.get("/secure", headers={"X-API-Key": "api-key-a"})
    second = client.get("/secure", headers={"X-API-Key": "api-key-a"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"]["code"] == "rate_limited"
    assert int(second.headers["Retry-After"]) >= 1


def test_jwt_auth_sets_tenant(monkeypatch) -> None:
    secret = "jwt-secret-test"
    token = _build_hs256_jwt(
        {
            "sub": "user-123",
            "tenant_id": "tenant_jwt",
            "scope": "read write",
        },
        secret=secret,
    )

    monkeypatch.setenv("API_SECURITY_ENABLED", "true")
    monkeypatch.setenv("API_RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("API_SECURITY_PUBLIC_PATHS", "/health")
    monkeypatch.setenv("JWT_SECRET_KEY", secret)
    monkeypatch.setenv("API_KEYS_JSON", "{}")
    _reset_security_cache()

    client = TestClient(_build_app())
    response = client.get(
        "/secure",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == "tenant_jwt"
    assert body["auth_type"] == "jwt"
