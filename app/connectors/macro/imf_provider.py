"""
app/connectors/macro/imf_provider.py

IMF provider placeholder for future integration.
"""

from __future__ import annotations

import requests

from app.config import ExternalHTTPSettings
from app.connectors.macro.base_provider import (
    BaseMacroProvider,
    MacroObservation,
    MacroProviderUnsupportedError,
)


class IMFMacroProvider(BaseMacroProvider):
    """
    Placeholder provider for IMF integration.
    """

    provider_name = "imf"

    def __init__(
        self,
        *,
        http_settings: ExternalHTTPSettings,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(source=self.provider_name, http_settings=http_settings, session=session)

    def fetch(
        self,
        *,
        country: str,
        metric: str,
        period_start: str | None = None,
        period_end: str | None = None,
        limit: int | None = None,
    ) -> list[MacroObservation]:
        _ = (country, metric, period_start, period_end, limit)
        raise MacroProviderUnsupportedError("imf: provider not implemented yet.")
