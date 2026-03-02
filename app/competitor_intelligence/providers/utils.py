"""Provider utilities shared across search API adapters."""

from __future__ import annotations

import asyncio
from urllib.parse import urlparse

import requests


class SearchProviderError(RuntimeError):
    """Raised when upstream provider calls fail."""


def normalize_domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.strip().lower()


async def request_json(
    *,
    method: str,
    url: str,
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
    params: dict[str, object] | None = None,
    json_payload: dict[str, object] | None = None,
) -> dict:
    """Issue blocking HTTP request in a thread and parse JSON body."""

    def _execute() -> dict:
        response = requests.request(
            method=method,
            url=url,
            timeout=timeout_seconds,
            headers=headers,
            params=params,
            json=json_payload,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise SearchProviderError("Search provider returned non-object JSON payload.")
        return payload

    try:
        return await asyncio.to_thread(_execute)
    except requests.HTTPError as exc:
        raise SearchProviderError(f"Search provider HTTP error: {exc}") from exc
    except requests.RequestException as exc:
        raise SearchProviderError(f"Search provider request failed: {exc}") from exc
    except ValueError as exc:
        raise SearchProviderError(f"Search provider returned invalid JSON: {exc}") from exc
