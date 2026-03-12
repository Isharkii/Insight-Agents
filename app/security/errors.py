from __future__ import annotations

from typing import Any, Mapping


class SecurityError(Exception):
    """
    Structured security exception used by middleware/auth dependencies.
    """

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        context: Mapping[str, Any] | None = None,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = str(code)
        self.message = str(message)
        self.context = dict(context or {})
        self.retry_after_seconds = retry_after_seconds
