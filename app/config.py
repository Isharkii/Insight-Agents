"""
app/config.py

Application-level configuration helpers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from db.config import load_env_files

_ALLOWED_APP_MODES = {"cloud"}


@lru_cache(maxsize=1)
def _load_env_once() -> None:
    """
    Ensure project `.env` files are loaded once before reading app settings.
    """

    load_env_files()


def _require_app_mode() -> str:
    """
    Read and validate APP_MODE from the environment.

    APP_MODE must be explicitly set to 'cloud'. Any other value—or the
    absence of the variable—raises RuntimeError to prevent silent local
    fallback behaviour.
    """

    _load_env_once()
    raw = os.getenv("APP_MODE")
    if raw is None:
        raise RuntimeError("APP_MODE must be explicitly set to 'cloud'.")
    mode = raw.strip().lower()
    if mode not in _ALLOWED_APP_MODES:
        raise RuntimeError(
            f"APP_MODE '{raw.strip()}' is not valid. "
            f"Allowed values: {sorted(_ALLOWED_APP_MODES)}."
        )
    return mode


@dataclass(frozen=True)
class AppSettings:
    """
    Top-level application mode settings.
    """

    mode: str


@lru_cache(maxsize=1)
def get_app_settings() -> AppSettings:
    """
    Return cached application settings.

    Raises RuntimeError if APP_MODE is missing or not set to 'cloud'.
    """

    return AppSettings(mode=_require_app_mode())


def _get_bool_env(name: str, default: bool) -> bool:
    """
    Read a boolean from environment variables with safe fallback.
    """

    _load_env_once()
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int_env(name: str, default: int) -> int:
    """
    Read an integer from environment variables with safe fallback.
    """

    _load_env_once()
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _get_float_env(name: str, default: float) -> float:
    """
    Read a float from environment variables with safe fallback.
    """

    _load_env_once()
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _get_str_env(name: str, default: str) -> str:
    """
    Read a string from environment variables with fallback.
    """

    _load_env_once()
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped if stripped else default


def _get_optional_str_env(name: str) -> str | None:
    """
    Read an optional string value from environment variables.
    """

    _load_env_once()
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


@dataclass(frozen=True)
class CSVIngestionSettings:
    """
    Runtime settings for CSV ingestion.
    """

    batch_size: int = 1000
    max_validation_errors: int = 500
    log_validation_errors: bool = True


@dataclass(frozen=True)
class ExternalHTTPSettings:
    """
    Shared HTTP behavior settings for external connectors.
    """

    timeout_seconds: float = 15.0
    max_retries: int = 3
    backoff_initial_seconds: float = 0.5
    backoff_multiplier: float = 2.0
    rate_limit_per_second: float = 5.0


@dataclass(frozen=True)
class ExternalIngestionSettings:
    """
    Runtime settings for external ingestion orchestration.
    """

    batch_size: int = 1000


@dataclass(frozen=True)
class NewsAPISettings:
    """
    News API connector settings.
    """

    enabled: bool = True
    api_key: str | None = None
    base_url: str = "https://newsapi.org/v2/everything"
    query: str = "global economy OR geopolitical OR climate OR energy"
    language: str = "en"
    page_size: int = 50
    region: str = "global"


@dataclass(frozen=True)
class GoogleTrendsSettings:
    """
    Google Trends connector settings.
    """

    enabled: bool = True
    rss_url: str = "https://trends.google.com/trending/rss"
    geo: str = "US"
    hl: str = "en-US"
    max_items: int = 50
    region: str = "US"


@dataclass(frozen=True)
class WorldBankSettings:
    """
    World Bank economic API connector settings.
    """

    enabled: bool = True
    base_url: str = "https://api.worldbank.org/v2"
    country_code: str = "WLD"
    indicator_code: str = "NY.GDP.MKTP.CD"
    per_page: int = 200
    latest_periods: int = 20


@lru_cache(maxsize=1)
def get_csv_ingestion_settings() -> CSVIngestionSettings:
    """
    Return cached CSV ingestion settings from environment variables.
    """

    return CSVIngestionSettings(
        batch_size=max(1, _get_int_env("CSV_INGEST_BATCH_SIZE", 1000)),
        max_validation_errors=max(1, _get_int_env("CSV_INGEST_MAX_VALIDATION_ERRORS", 500)),
        log_validation_errors=_get_bool_env("CSV_INGEST_LOG_VALIDATION_ERRORS", True),
    )


@lru_cache(maxsize=1)
def get_external_http_settings() -> ExternalHTTPSettings:
    """
    Return shared connector HTTP settings from environment variables.
    """

    return ExternalHTTPSettings(
        timeout_seconds=max(1.0, _get_float_env("EXTERNAL_HTTP_TIMEOUT_SECONDS", 15.0)),
        max_retries=max(0, _get_int_env("EXTERNAL_HTTP_MAX_RETRIES", 3)),
        backoff_initial_seconds=max(0.1, _get_float_env("EXTERNAL_HTTP_BACKOFF_INITIAL_SECONDS", 0.5)),
        backoff_multiplier=max(1.0, _get_float_env("EXTERNAL_HTTP_BACKOFF_MULTIPLIER", 2.0)),
        rate_limit_per_second=max(0.1, _get_float_env("EXTERNAL_HTTP_RATE_LIMIT_PER_SECOND", 5.0)),
    )


@lru_cache(maxsize=1)
def get_external_ingestion_settings() -> ExternalIngestionSettings:
    """
    Return external ingestion orchestration settings.
    """

    return ExternalIngestionSettings(
        batch_size=max(1, _get_int_env("EXTERNAL_INGEST_BATCH_SIZE", 1000)),
    )


@lru_cache(maxsize=1)
def get_news_api_settings() -> NewsAPISettings:
    """
    Return News API connector settings from environment variables.
    """

    return NewsAPISettings(
        enabled=_get_bool_env("NEWS_API_ENABLED", True),
        api_key=_get_optional_str_env("NEWS_API_KEY"),
        base_url=_get_str_env("NEWS_API_BASE_URL", "https://newsapi.org/v2/everything"),
        query=_get_str_env("NEWS_API_QUERY", "global economy OR geopolitical OR climate OR energy"),
        language=_get_str_env("NEWS_API_LANGUAGE", "en"),
        page_size=max(1, _get_int_env("NEWS_API_PAGE_SIZE", 50)),
        region=_get_str_env("NEWS_API_REGION", "global"),
    )


@lru_cache(maxsize=1)
def get_google_trends_settings() -> GoogleTrendsSettings:
    """
    Return Google Trends connector settings from environment variables.
    """

    return GoogleTrendsSettings(
        enabled=_get_bool_env("GOOGLE_TRENDS_ENABLED", True),
        rss_url=_get_str_env("GOOGLE_TRENDS_RSS_URL", "https://trends.google.com/trending/rss"),
        geo=_get_str_env("GOOGLE_TRENDS_GEO", "US"),
        hl=_get_str_env("GOOGLE_TRENDS_HL", "en-US"),
        max_items=max(1, _get_int_env("GOOGLE_TRENDS_MAX_ITEMS", 50)),
        region=_get_str_env("GOOGLE_TRENDS_REGION", _get_str_env("GOOGLE_TRENDS_GEO", "US")),
    )


@lru_cache(maxsize=1)
def get_world_bank_settings() -> WorldBankSettings:
    """
    Return World Bank connector settings from environment variables.
    """

    return WorldBankSettings(
        enabled=_get_bool_env("WORLD_BANK_ENABLED", True),
        base_url=_get_str_env("WORLD_BANK_BASE_URL", "https://api.worldbank.org/v2"),
        country_code=_get_str_env("WORLD_BANK_COUNTRY_CODE", "WLD"),
        indicator_code=_get_str_env("WORLD_BANK_INDICATOR_CODE", "NY.GDP.MKTP.CD"),
        per_page=max(1, _get_int_env("WORLD_BANK_PER_PAGE", 200)),
        latest_periods=max(1, _get_int_env("WORLD_BANK_LATEST_PERIODS", 20)),
    )
