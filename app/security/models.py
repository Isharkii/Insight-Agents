from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SecurityContext:
    """
    Request-scoped security context propagated through FastAPI dependencies.
    """

    request_id: str
    tenant_id: str
    subject: str
    auth_type: str
    scopes: frozenset[str] = field(default_factory=frozenset)
    claims: dict[str, Any] = field(default_factory=dict)
    allowed_entities: frozenset[str] = field(default_factory=frozenset)

    def has_scope(self, scope: str) -> bool:
        if not scope:
            return True
        if "*" in self.scopes:
            return True
        return scope in self.scopes
