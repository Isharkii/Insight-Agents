"""
app/connectors/macro/registry.py

Registry/factory for swappable macro providers.
"""

from __future__ import annotations

from typing import Mapping

from app.config import ExternalHTTPSettings
from app.connectors.macro.base_provider import BaseMacroProvider, MacroProviderUnsupportedError
from app.connectors.macro.fred_provider import FREDMacroProvider
from app.connectors.macro.imf_provider import IMFMacroProvider
from app.connectors.macro.world_bank_provider import WorldBankMacroProvider


class MacroProviderRegistry:
    """
    In-memory provider registry keyed by provider name.
    """

    def __init__(self, providers: Mapping[str, BaseMacroProvider] | None = None) -> None:
        self._providers: dict[str, BaseMacroProvider] = {}
        for name, provider in (providers or {}).items():
            self.register(name, provider)

    def register(self, name: str, provider: BaseMacroProvider) -> None:
        normalized = self._normalize_name(name)
        self._providers[normalized] = provider

    def get(self, name: str) -> BaseMacroProvider:
        normalized = self._normalize_name(name)
        provider = self._providers.get(normalized)
        if provider is None:
            available = ", ".join(sorted(self._providers.keys()))
            raise MacroProviderUnsupportedError(
                f"Unsupported macro provider '{name}'. Available: {available}."
            )
        return provider

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers.keys()))

    @staticmethod
    def _normalize_name(name: str) -> str:
        return str(name or "").strip().lower()


def build_default_macro_provider_registry(
    *,
    http_settings: ExternalHTTPSettings,
    fred_api_key: str | None = None,
    fred_base_url: str = "https://api.stlouisfed.org/fred/series/observations",
    world_bank_base_url: str = "https://api.worldbank.org/v2",
    world_bank_per_page: int = 200,
    world_bank_latest_periods: int = 20,
) -> MacroProviderRegistry:
    registry = MacroProviderRegistry()
    registry.register(
        "world_bank",
        WorldBankMacroProvider(
            http_settings=http_settings,
            base_url=world_bank_base_url,
            per_page=world_bank_per_page,
            latest_periods=world_bank_latest_periods,
        ),
    )
    registry.register(
        "fred",
        FREDMacroProvider(
            http_settings=http_settings,
            api_key=fred_api_key,
            base_url=fred_base_url,
        ),
    )
    registry.register(
        "imf",
        IMFMacroProvider(http_settings=http_settings),
    )
    return registry


def get_macro_provider(
    provider_name: str,
    *,
    http_settings: ExternalHTTPSettings,
    fred_api_key: str | None = None,
    fred_base_url: str = "https://api.stlouisfed.org/fred/series/observations",
    world_bank_base_url: str = "https://api.worldbank.org/v2",
    world_bank_per_page: int = 200,
    world_bank_latest_periods: int = 20,
) -> BaseMacroProvider:
    registry = build_default_macro_provider_registry(
        http_settings=http_settings,
        fred_api_key=fred_api_key,
        fred_base_url=fred_base_url,
        world_bank_base_url=world_bank_base_url,
        world_bank_per_page=world_bank_per_page,
        world_bank_latest_periods=world_bank_latest_periods,
    )
    return registry.get(provider_name)

