"""
Environment + JSON config loader for competitor scraping.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin

from db.config import load_env_files

from app.scraping.config.models import CompetitorDomainConfig, CompetitorScrapingSettings


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_str_env(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    stripped = raw.strip()
    return stripped if stripped else default


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_config_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (_project_root() / candidate).resolve()


@lru_cache(maxsize=1)
def get_competitor_scraping_settings() -> CompetitorScrapingSettings:
    """
    Return cached scraper settings from environment variables.
    """

    load_env_files()
    config_path = _get_str_env(
        "COMPETITOR_SCRAPE_CONFIG_PATH",
        "app/scraping/config/competitors.json",
    )
    return CompetitorScrapingSettings(
        config_path=str(_resolve_config_path(config_path)),
        default_user_agent=_get_str_env(
            "COMPETITOR_SCRAPE_USER_AGENT",
            "InsightAgentBot/1.0 (+https://example.com/bot)",
        ),
        default_rate_limit_per_second=max(
            0.1,
            _get_float_env("COMPETITOR_SCRAPE_RATE_LIMIT_PER_SECOND", 1.0),
        ),
        timeout_seconds=max(
            1.0,
            _get_float_env("COMPETITOR_SCRAPE_TIMEOUT_SECONDS", 15.0),
        ),
        max_retries=max(
            0,
            _get_int_env("COMPETITOR_SCRAPE_MAX_RETRIES", 3),
        ),
        backoff_initial_seconds=max(
            0.1,
            _get_float_env("COMPETITOR_SCRAPE_BACKOFF_INITIAL_SECONDS", 0.5),
        ),
        backoff_multiplier=max(
            1.0,
            _get_float_env("COMPETITOR_SCRAPE_BACKOFF_MULTIPLIER", 2.0),
        ),
        allow_when_robots_unreachable=_get_bool_env(
            "COMPETITOR_SCRAPE_ALLOW_WHEN_ROBOTS_UNREACHABLE",
            True,
        ),
        storage_batch_size=max(
            1,
            _get_int_env("COMPETITOR_SCRAPE_STORAGE_BATCH_SIZE", 1000),
        ),
    )


def load_competitor_configs(*, config_path: str) -> list[CompetitorDomainConfig]:
    """
    Load competitor configurations from a JSON file.
    """

    path = _resolve_config_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Competitor config file not found: {path}")

    raw_data = json.loads(path.read_text(encoding="utf-8"))
    competitors = raw_data.get("competitors", [])
    if not isinstance(competitors, list):
        raise ValueError("Invalid competitor config: 'competitors' must be a list.")

    parsed: list[CompetitorDomainConfig] = []
    for entry in competitors:
        if not isinstance(entry, dict):
            continue

        name = str(entry.get("name", "")).strip()
        base_url = str(entry.get("base_url", "")).strip()
        if not name or not base_url:
            continue

        pages = _normalize_pages(base_url=base_url, pages=entry.get("pages", {}))
        selectors = _normalize_selectors(entry.get("selectors", {}))

        parsed.append(
            CompetitorDomainConfig(
                name=name,
                entity_name=str(entry.get("entity_name", name)).strip() or name,
                scraper_type=str(entry.get("scraper_type", "configurable")).strip().lower(),
                base_url=base_url.rstrip("/"),
                pages=pages,
                selectors=selectors,
                enabled=_optional_bool(entry.get("enabled"), True),
                user_agent=_optional_str(entry.get("user_agent")),
                rate_limit_per_second=_optional_float(entry.get("rate_limit_per_second")),
                region=_optional_str(entry.get("region")),
                headers=_normalize_headers(entry.get("headers", {})),
                scraper_class=_optional_str(entry.get("scraper_class")),
            )
        )

    return parsed


def _normalize_pages(*, base_url: str, pages: object) -> dict[str, str]:
    if not isinstance(pages, dict):
        return {}

    normalized: dict[str, str] = {}
    for key, value in pages.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        page_kind = key.strip().lower()
        raw_url = value.strip()
        if not page_kind or not raw_url:
            continue
        if raw_url.startswith(("http://", "https://")):
            normalized[page_kind] = raw_url
        else:
            normalized[page_kind] = urljoin(f"{base_url.rstrip('/')}/", raw_url.lstrip("/"))
    return normalized


def _normalize_selectors(selectors: object) -> dict[str, list[str]]:
    if not isinstance(selectors, dict):
        return {}

    normalized: dict[str, list[str]] = {}
    for key, value in selectors.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, str):
            selector_list = [value.strip()] if value.strip() else []
        elif isinstance(value, list):
            selector_list = [
                item.strip()
                for item in value
                if isinstance(item, str) and item.strip()
            ]
        else:
            selector_list = []
        normalized[key.strip().lower()] = selector_list
    return normalized


def _normalize_headers(headers: object) -> dict[str, str]:
    if not isinstance(headers, dict):
        return {}

    normalized: dict[str, str] = {}
    for key, value in headers.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if key.strip() and value.strip():
            normalized[key.strip()] = value.strip()
    return normalized


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default
