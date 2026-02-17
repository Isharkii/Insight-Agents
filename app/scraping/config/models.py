"""
Scraping configuration models.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CompetitorDomainConfig:
    """
    One competitor scrape target configuration.
    """

    name: str
    entity_name: str
    scraper_type: str
    base_url: str
    pages: dict[str, str]
    selectors: dict[str, list[str]] = field(default_factory=dict)
    enabled: bool = True
    user_agent: str | None = None
    rate_limit_per_second: float | None = None
    region: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    scraper_class: str | None = None


@dataclass(frozen=True)
class CompetitorScrapingSettings:
    """
    Runtime settings for competitor scraping.
    """

    config_path: str
    default_user_agent: str
    default_rate_limit_per_second: float
    timeout_seconds: float
    max_retries: int
    backoff_initial_seconds: float
    backoff_multiplier: float
    allow_when_robots_unreachable: bool
    storage_batch_size: int
