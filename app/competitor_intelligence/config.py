"""Configuration model for competitor intelligence components."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: list[str] | None = None) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default or [])
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


class SearchProviderApiConfig(BaseModel):
    """Credentials and endpoint for one search provider."""

    model_config = ConfigDict(extra="forbid")

    endpoint: str = Field(min_length=1)
    api_key: str = ""
    timeout_seconds: float = Field(default=12.0, ge=1.0, le=120.0)


class ScraperConfig(BaseModel):
    """HTTP scraper settings."""

    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float = Field(default=15.0, ge=1.0, le=120.0)
    user_agent: str = Field(min_length=1, max_length=300)
    max_text_chars: int = Field(default=15000, ge=500, le=500000)
    max_concurrency: int = Field(default=6, ge=1, le=32)
    domain_blacklist: list[str] = Field(default_factory=list)


class CacheConfig(BaseModel):
    """In-memory cache settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    ttl_seconds: int = Field(default=600, ge=1, le=86400)
    max_size: int = Field(default=4096, ge=64, le=200000)


class ExtractionConfig(BaseModel):
    """Extractor strategy and limits."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["deterministic", "llm_structured"] = "deterministic"
    max_docs: int = Field(default=6, ge=1, le=30)
    max_chars_per_doc: int = Field(default=3000, ge=300, le=30000)


class CompetitorIntelligenceConfig(BaseModel):
    """Top-level module configuration."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["brave", "serper", "tavily"] = "brave"
    search_max_concurrency: int = Field(default=12, ge=1, le=100)
    max_pages: int = Field(default=3, ge=1, le=20)
    query_templates: list[str] = Field(
        default_factory=lambda: [
            "{competitor} SaaS pricing",
            "{competitor} churn rate",
            "{competitor} customer growth",
            "{competitor} product updates",
        ]
    )
    brave: SearchProviderApiConfig = Field(
        default_factory=lambda: SearchProviderApiConfig(
            endpoint="https://api.search.brave.com/res/v1/web/search",
            api_key="",
            timeout_seconds=12.0,
        )
    )
    serper: SearchProviderApiConfig = Field(
        default_factory=lambda: SearchProviderApiConfig(
            endpoint="https://google.serper.dev/search",
            api_key="",
            timeout_seconds=12.0,
        )
    )
    tavily: SearchProviderApiConfig = Field(
        default_factory=lambda: SearchProviderApiConfig(
            endpoint="https://api.tavily.com/search",
            api_key="",
            timeout_seconds=12.0,
        )
    )
    scraper: ScraperConfig = Field(
        default_factory=lambda: ScraperConfig(
            timeout_seconds=15.0,
            user_agent="InsightAgentCompetitorIntel/1.0",
            max_text_chars=15000,
            max_concurrency=6,
        )
    )
    cache: CacheConfig = Field(
        default_factory=lambda: CacheConfig(
            enabled=True,
            ttl_seconds=600,
            max_size=4096,
        )
    )
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)

    @classmethod
    def from_env(cls) -> "CompetitorIntelligenceConfig":
        """Build settings from environment without hardcoded runtime coupling."""
        return cls(
            provider=os.getenv("COMP_INTEL_PROVIDER", "brave").strip().lower() or "brave",
            search_max_concurrency=_env_int("COMP_INTEL_SEARCH_MAX_CONCURRENCY", 12),
            max_pages=_env_int("COMP_INTEL_MAX_PAGES", 3),
            brave=SearchProviderApiConfig(
                endpoint=os.getenv(
                    "COMP_INTEL_BRAVE_ENDPOINT",
                    "https://api.search.brave.com/res/v1/web/search",
                ).strip(),
                api_key=os.getenv("COMP_INTEL_BRAVE_API_KEY", "").strip(),
                timeout_seconds=_env_float("COMP_INTEL_BRAVE_TIMEOUT_SECONDS", 12.0),
            ),
            serper=SearchProviderApiConfig(
                endpoint=os.getenv("COMP_INTEL_SERPER_ENDPOINT", "https://google.serper.dev/search").strip(),
                api_key=os.getenv("COMP_INTEL_SERPER_API_KEY", "").strip(),
                timeout_seconds=_env_float("COMP_INTEL_SERPER_TIMEOUT_SECONDS", 12.0),
            ),
            tavily=SearchProviderApiConfig(
                endpoint=os.getenv("COMP_INTEL_TAVILY_ENDPOINT", "https://api.tavily.com/search").strip(),
                api_key=os.getenv("COMP_INTEL_TAVILY_API_KEY", "").strip(),
                timeout_seconds=_env_float("COMP_INTEL_TAVILY_TIMEOUT_SECONDS", 12.0),
            ),
            query_templates=[
                item.strip()
                for item in os.getenv(
                    "COMP_INTEL_QUERY_TEMPLATES",
                    "{competitor} SaaS pricing|{competitor} churn rate|{competitor} customer growth|{competitor} product updates",
                ).split("|")
                if item.strip()
            ],
            scraper=ScraperConfig(
                timeout_seconds=_env_float("COMP_INTEL_SCRAPER_TIMEOUT_SECONDS", 15.0),
                user_agent=os.getenv("COMP_INTEL_SCRAPER_USER_AGENT", "InsightAgentCompetitorIntel/1.0").strip(),
                max_text_chars=_env_int("COMP_INTEL_SCRAPER_MAX_TEXT_CHARS", 15000),
                max_concurrency=_env_int("COMP_INTEL_SCRAPER_MAX_CONCURRENCY", 6),
                domain_blacklist=_env_csv("COMP_INTEL_DOMAIN_BLACKLIST", []),
            ),
            cache=CacheConfig(
                enabled=_env_bool("COMP_INTEL_CACHE_ENABLED", True),
                ttl_seconds=_env_int("COMP_INTEL_CACHE_TTL_SECONDS", 600),
                max_size=_env_int("COMP_INTEL_CACHE_MAX_SIZE", 4096),
            ),
            extraction=ExtractionConfig(
                mode=os.getenv("COMP_INTEL_EXTRACTION_MODE", "deterministic").strip().lower() or "deterministic",
                max_docs=_env_int("COMP_INTEL_EXTRACTION_MAX_DOCS", 6),
                max_chars_per_doc=_env_int("COMP_INTEL_EXTRACTION_MAX_CHARS_PER_DOC", 3000),
            ),
        )
