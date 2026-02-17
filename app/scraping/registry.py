"""
Configurable scraper class registry and factory.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping

import requests

from app.scraping.base import ScraperBase
from app.scraping.config.models import CompetitorDomainConfig, CompetitorScrapingSettings
from app.scraping.normalization import CanonicalNormalizer
from app.scraping.rate_limiter import DomainRateLimiter
from app.scraping.robots import RobotsPolicyManager
from app.scraping.scrapers import ConfigurableCompetitorScraper, SaaSCompetitorScraper


class ScraperRegistry:
    """
    Scraper registry supporting built-ins and dynamic import paths.
    """

    def __init__(self, registrations: Mapping[str, type[ScraperBase]] | None = None) -> None:
        builtins: dict[str, type[ScraperBase]] = {
            "configurable": ConfigurableCompetitorScraper,
            "saas": SaaSCompetitorScraper,
        }
        if registrations:
            builtins.update(registrations)
        self._registrations = builtins

    def register(self, *, scraper_type: str, scraper_class: type[ScraperBase]) -> None:
        self._registrations[scraper_type.strip().lower()] = scraper_class

    def create_scraper(
        self,
        *,
        config: CompetitorDomainConfig,
        settings: CompetitorScrapingSettings,
        session: requests.Session,
        robots_policy: RobotsPolicyManager,
        rate_limiter: DomainRateLimiter,
        normalizer: CanonicalNormalizer,
    ) -> ScraperBase:
        scraper_class = self._resolve_scraper_class(config)
        return scraper_class(
            config=config,
            settings=settings,
            session=session,
            robots_policy=robots_policy,
            rate_limiter=rate_limiter,
            normalizer=normalizer,
        )

    def _resolve_scraper_class(self, config: CompetitorDomainConfig) -> type[ScraperBase]:
        if config.scraper_class:
            return self._load_dynamic_class(config.scraper_class)

        resolved = self._registrations.get(config.scraper_type)
        if resolved is None:
            allowed = ", ".join(sorted(self._registrations.keys()))
            raise ValueError(
                f"Unknown scraper_type='{config.scraper_type}' for competitor='{config.name}'. "
                f"Allowed types: {allowed}."
            )
        return resolved

    @staticmethod
    def _load_dynamic_class(path: str) -> type[ScraperBase]:
        if ":" not in path:
            raise ValueError(f"Invalid scraper_class '{path}'. Use 'module.path:ClassName'.")

        module_path, class_name = path.split(":", 1)
        module = importlib.import_module(module_path)
        loaded = getattr(module, class_name, None)
        if loaded is None:
            raise ValueError(f"Unable to resolve scraper class '{path}'.")
        if not isinstance(loaded, type) or not issubclass(loaded, ScraperBase):
            raise ValueError(f"Class '{path}' must inherit from ScraperBase.")
        return loaded
