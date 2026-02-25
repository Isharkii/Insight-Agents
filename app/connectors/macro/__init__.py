"""
Macro provider abstraction exports.
"""

from app.connectors.macro.base_provider import (
    BaseMacroProvider,
    MacroObservation,
    MacroProviderConfigurationError,
    MacroProviderError,
    MacroProviderRateLimitError,
    MacroProviderRequestError,
    MacroProviderResponseError,
    MacroProviderUnsupportedError,
)
from app.connectors.macro.fred_provider import FREDMacroProvider
from app.connectors.macro.imf_provider import IMFMacroProvider
from app.connectors.macro.registry import (
    MacroProviderRegistry,
    build_default_macro_provider_registry,
    get_macro_provider,
)
from app.connectors.macro.world_bank_provider import WorldBankMacroProvider

__all__ = [
    "BaseMacroProvider",
    "MacroObservation",
    "MacroProviderError",
    "MacroProviderConfigurationError",
    "MacroProviderRequestError",
    "MacroProviderRateLimitError",
    "MacroProviderResponseError",
    "MacroProviderUnsupportedError",
    "WorldBankMacroProvider",
    "FREDMacroProvider",
    "IMFMacroProvider",
    "MacroProviderRegistry",
    "build_default_macro_provider_registry",
    "get_macro_provider",
]

