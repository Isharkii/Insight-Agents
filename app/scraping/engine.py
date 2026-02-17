"""
Competitor scraping engine.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import requests

from app.domain.competitor_scraping import CompetitorScrapeSummary
from app.scraping.config import load_competitor_configs
from app.scraping.config.models import CompetitorDomainConfig, CompetitorScrapingSettings
from app.scraping.logging_utils import log_event
from app.scraping.normalization import CanonicalNormalizer
from app.scraping.rate_limiter import DomainRateLimiter
from app.scraping.registry import ScraperRegistry
from app.scraping.robots import RobotsPolicyManager
from app.scraping.storage import InsightStorage

logger = logging.getLogger(__name__)


class CompetitorScrapingEngine:
    """
    Orchestrates config-driven competitor scraping and persistence.
    """

    def __init__(
        self,
        *,
        settings: CompetitorScrapingSettings,
        storage: InsightStorage,
        registry: ScraperRegistry | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._registry = registry or ScraperRegistry()
        self._session = session or requests.Session()

    def run(self, *, competitors: Sequence[str] | None = None) -> list[CompetitorScrapeSummary]:
        configs = load_competitor_configs(config_path=self._settings.config_path)
        selected = self._select_competitors(configs=configs, competitors=competitors)
        if not selected:
            raise ValueError("No enabled competitors matched the run criteria.")

        robots_policy = RobotsPolicyManager(
            session=self._session,
            timeout_seconds=self._settings.timeout_seconds,
            allow_when_unreachable=self._settings.allow_when_robots_unreachable,
        )
        rate_limiter = DomainRateLimiter(
            default_rate_limit_per_second=self._settings.default_rate_limit_per_second
        )
        normalizer = CanonicalNormalizer()

        summaries: list[CompetitorScrapeSummary] = []
        for config in selected:
            try:
                scraper = self._registry.create_scraper(
                    config=config,
                    settings=self._settings,
                    session=self._session,
                    robots_policy=robots_policy,
                    rate_limiter=rate_limiter,
                    normalizer=normalizer,
                )
                run_result = scraper.scrape()
                inserted = self._storage.store(run_result.records)
                status = "success"
                if run_result.failed_pages > 0:
                    status = "partial_success" if inserted > 0 else "failed"
                summary = CompetitorScrapeSummary(
                    competitor=config.name,
                    records_scraped=len(run_result.records),
                    records_inserted=inserted,
                    failed_pages=run_result.failed_pages,
                    status=status,
                    errors=run_result.errors,
                )
                summaries.append(summary)
                log_event(
                    logger,
                    logging.INFO,
                    "competitor_scrape_completed",
                    competitor=config.name,
                    records_scraped=summary.records_scraped,
                    records_inserted=summary.records_inserted,
                    failed_pages=summary.failed_pages,
                    status=summary.status,
                )
            except Exception as exc:
                message = str(exc)
                summaries.append(
                    CompetitorScrapeSummary(
                        competitor=config.name,
                        records_scraped=0,
                        records_inserted=0,
                        failed_pages=max(1, len(config.pages)),
                        status="failed",
                        errors=[message],
                    )
                )
                log_event(
                    logger,
                    logging.ERROR,
                    "competitor_scrape_failed",
                    competitor=config.name,
                    error=message,
                )
        return summaries

    @staticmethod
    def _select_competitors(
        *,
        configs: list[CompetitorDomainConfig],
        competitors: Sequence[str] | None,
    ) -> list[CompetitorDomainConfig]:
        enabled = [config for config in configs if config.enabled]
        if not competitors:
            return enabled

        normalized = {item.strip().lower() for item in competitors if item.strip()}
        if not normalized:
            return enabled
        return [config for config in enabled if config.name.lower() in normalized]
