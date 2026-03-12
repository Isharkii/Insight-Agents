from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache

from db.config import load_env_files


def _load_env() -> None:
    load_env_files()


def _env_bool(name: str, default: bool) -> bool:
    _load_env()
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    _load_env()
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_str(name: str, default: str = "") -> str:
    _load_env()
    raw = os.getenv(name)
    if raw is None:
        return default
    stripped = raw.strip()
    return stripped if stripped else default


def _env_optional_str(name: str) -> str | None:
    value = _env_str(name, "")
    return value or None


def _parse_security_api_keys(raw: str | None) -> tuple[dict[str, str], dict[str, frozenset[str]]]:
    """
    Parse API key config from env.

    Supported formats for API_KEYS_JSON:
    1) {"tenant_a": "key-a", "tenant_b": "key-b"}
    2) {
         "tenant_a": {"key": "key-a", "entities": ["acme", "beta"]},
         "tenant_b": {"key": "key-b"}
       }

    Returns:
        - key_to_tenant map
        - tenant_to_allowed_entities map
    """
    key_to_tenant: dict[str, str] = {}
    tenant_entities: dict[str, frozenset[str]] = {}
    if not raw:
        return key_to_tenant, tenant_entities

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return key_to_tenant, tenant_entities
    if not isinstance(payload, dict):
        return key_to_tenant, tenant_entities

    for tenant_id_raw, value in payload.items():
        tenant_id = str(tenant_id_raw or "").strip()
        if not tenant_id:
            continue
        if isinstance(value, str):
            api_key = value.strip()
            entities = frozenset()
        elif isinstance(value, dict):
            api_key = str(value.get("key") or "").strip()
            raw_entities = value.get("entities")
            if isinstance(raw_entities, list):
                entities = frozenset(
                    str(item).strip().lower()
                    for item in raw_entities
                    if str(item).strip()
                )
            else:
                entities = frozenset()
        else:
            continue

        if not api_key:
            continue
        key_to_tenant[api_key] = tenant_id
        if entities:
            tenant_entities[tenant_id] = entities

    return key_to_tenant, tenant_entities


def _parse_public_paths(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return (
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/dashboard",
            "/dashboard/assets",
        )
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        return tuple()
    return tuple(values)


@dataclass(frozen=True)
class SecuritySettings:
    enabled: bool
    rate_limit_enabled: bool
    rate_limit_window_seconds: int
    rate_limit_max_requests: int
    jwt_secret_key: str | None
    jwt_issuer: str | None
    jwt_audience: str | None
    key_to_tenant: dict[str, str]
    tenant_entities: dict[str, frozenset[str]]
    legacy_api_key: str | None
    public_path_prefixes: tuple[str, ...]
    tenant_header_name: str = "X-Tenant-ID"
    request_id_header_name: str = "X-Request-ID"
    api_key_header_name: str = "X-API-Key"
    authorization_header_name: str = "Authorization"


@lru_cache(maxsize=1)
def get_security_settings() -> SecuritySettings:
    key_to_tenant, tenant_entities = _parse_security_api_keys(
        _env_optional_str("API_KEYS_JSON")
    )
    return SecuritySettings(
        enabled=_env_bool("API_SECURITY_ENABLED", True),
        rate_limit_enabled=_env_bool("API_RATE_LIMIT_ENABLED", True),
        rate_limit_window_seconds=max(1, _env_int("API_RATE_LIMIT_WINDOW_SECONDS", 60)),
        rate_limit_max_requests=max(1, _env_int("API_RATE_LIMIT_MAX_REQUESTS", 120)),
        jwt_secret_key=_env_optional_str("JWT_SECRET_KEY"),
        jwt_issuer=_env_optional_str("JWT_ISSUER"),
        jwt_audience=_env_optional_str("JWT_AUDIENCE"),
        key_to_tenant=key_to_tenant,
        tenant_entities=tenant_entities,
        legacy_api_key=_env_optional_str("DECISION_ENGINE_API_KEY")
        or _env_optional_str("DECISION_API_KEY"),
        public_path_prefixes=_parse_public_paths(os.getenv("API_SECURITY_PUBLIC_PATHS")),
    )
