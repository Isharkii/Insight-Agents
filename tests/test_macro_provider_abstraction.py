from __future__ import annotations

from typing import Any

import pytest
import requests

from app.config import ExternalHTTPSettings
from app.connectors.macro import (
    BaseMacroProvider,
    FREDMacroProvider,
    IMFMacroProvider,
    MacroProviderRequestError,
    MacroProviderUnsupportedError,
    WorldBankMacroProvider,
    build_default_macro_provider_registry,
)


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        payload: Any = None,
        headers: dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text

    def json(self) -> Any:
        return self._payload


class _FakeSession:
    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def request(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if not self._outcomes:
            raise AssertionError("No remaining fake outcomes.")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _CustomProvider(BaseMacroProvider):
    provider_name = "custom"

    def fetch(
        self,
        *,
        country: str,
        metric: str,
        period_start: str | None = None,
        period_end: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        _ = (period_start, period_end, limit)
        observation = self._build_observation(
            country=country,
            metric=metric,
            period_start="2026-01-01",
            period_end="2026-01-31",
            value=1.0,
        )
        return [observation] if observation else []


def test_world_bank_provider_normalizes_observation_contract() -> None:
    session = _FakeSession(
        [
            _FakeResponse(
                status_code=200,
                payload=[
                    {"page": 1, "pages": 1},
                    [
                        {"date": "2024", "value": 111.2},
                        {"date": "2023", "value": None},
                    ],
                ],
            )
        ]
    )
    provider = WorldBankMacroProvider(
        http_settings=ExternalHTTPSettings(rate_limit_per_second=0.0),
        session=session,  # type: ignore[arg-type]
    )

    rows = provider.fetch(country="us", metric="NY.GDP.MKTP.CD", limit=5)
    assert len(rows) == 1
    assert rows[0] == {
        "country": "US",
        "metric": "NY.GDP.MKTP.CD",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31",
        "value": 111.2,
        "source": "world_bank",
    }


def test_fred_provider_retries_rate_limit_and_returns_standardized_rows(monkeypatch) -> None:
    session = _FakeSession(
        [
            _FakeResponse(status_code=429, headers={"Retry-After": "1"}),
            _FakeResponse(
                status_code=200,
                payload={
                    "observations": [
                        {"date": "2025-01-01", "value": "100.0"},
                        {"date": "2025-02-01", "value": "."},
                    ]
                },
            ),
        ]
    )
    sleeps: list[float] = []
    monkeypatch.setattr(
        "app.connectors.macro.base_provider.time.sleep",
        lambda seconds: sleeps.append(float(seconds)),
    )

    provider = FREDMacroProvider(
        http_settings=ExternalHTTPSettings(
            timeout_seconds=1.0,
            max_retries=2,
            backoff_initial_seconds=0.1,
            backoff_multiplier=2.0,
            rate_limit_per_second=0.0,
        ),
        api_key="test-key",
        session=session,  # type: ignore[arg-type]
    )

    rows = provider.fetch(
        country="us",
        metric="CPIAUCSL",
        period_start="2025-01-01",
        period_end="2025-12-31",
        limit=10,
    )
    assert len(rows) == 1
    assert rows[0] == {
        "country": "US",
        "metric": "CPIAUCSL",
        "period_start": "2025-01-01",
        "period_end": "2025-01-01",
        "value": 100.0,
        "source": "fred",
    }
    assert len(session.calls) == 2
    assert 1.0 in sleeps


def test_registry_supports_swappable_providers_and_future_imf_provider() -> None:
    http_settings = ExternalHTTPSettings(rate_limit_per_second=0.0)
    registry = build_default_macro_provider_registry(
        http_settings=http_settings,
        fred_api_key="x",
    )
    assert isinstance(registry.get("world_bank"), WorldBankMacroProvider)
    assert isinstance(registry.get("fred"), FREDMacroProvider)
    assert isinstance(registry.get("imf"), IMFMacroProvider)

    custom_provider = _CustomProvider(
        source="custom",
        http_settings=http_settings,
    )
    registry.register("custom", custom_provider)
    assert registry.get("custom") is custom_provider

    with pytest.raises(MacroProviderUnsupportedError):
        registry.get("missing")

    with pytest.raises(MacroProviderUnsupportedError):
        registry.get("imf").fetch(country="US", metric="NGDP_RPCH")


def test_request_error_after_retry_exhaustion() -> None:
    session = _FakeSession([requests.ConnectionError("network down"), requests.ConnectionError("network down")])
    provider = _CustomProvider(
        source="custom",
        http_settings=ExternalHTTPSettings(
            max_retries=1,
            backoff_initial_seconds=0.0,
            rate_limit_per_second=0.0,
        ),
        session=session,  # type: ignore[arg-type]
    )

    with pytest.raises(MacroProviderRequestError):
        provider._request_json(method="GET", url="https://example.com")
