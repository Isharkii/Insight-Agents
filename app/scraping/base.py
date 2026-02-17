"""
Base scraper abstraction for competitor scraping.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

from app.scraping.config.models import CompetitorDomainConfig, CompetitorScrapingSettings
from app.scraping.logging_utils import log_event
from app.scraping.normalization import CanonicalNormalizer
from app.scraping.rate_limiter import DomainRateLimiter
from app.scraping.robots import RobotsPolicyManager
from app.scraping.types import ParsedCompetitorData, ScraperRunResult

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class ScraperBase(ABC):
    """
    Base class implementing compliant fetch mechanics and scrape orchestration.
    """

    def __init__(
        self,
        *,
        config: CompetitorDomainConfig,
        settings: CompetitorScrapingSettings,
        session: requests.Session,
        robots_policy: RobotsPolicyManager,
        rate_limiter: DomainRateLimiter,
        normalizer: CanonicalNormalizer,
    ) -> None:
        self.config = config
        self.settings = settings
        self.session = session
        self.robots_policy = robots_policy
        self.rate_limiter = rate_limiter
        self.normalizer = normalizer

        self.user_agent = config.user_agent or settings.default_user_agent
        self.request_headers = {"User-Agent": self.user_agent, **config.headers}

    def scrape(self) -> ScraperRunResult:
        """
        Scrape configured pages and return normalized canonical records.
        """

        parsed = ParsedCompetitorData()
        failed_pages = 0
        errors: list[str] = []

        for page_kind, page_url in self.page_plan().items():
            try:
                if not self.robots_policy.can_fetch(url=page_url, user_agent=self.user_agent):
                    reason = f"Blocked by robots.txt page={page_kind} url={page_url}"
                    errors.append(reason)
                    failed_pages += 1
                    log_event(
                        logger,
                        logging.WARNING,
                        "page_blocked_by_robots",
                        competitor=self.config.name,
                        page_kind=page_kind,
                        page_url=page_url,
                    )
                    continue

                crawl_delay = self.robots_policy.crawl_delay(
                    url=page_url,
                    user_agent=self.user_agent,
                )
                self.rate_limiter.wait(
                    url=page_url,
                    rate_limit_per_second=self.config.rate_limit_per_second,
                    crawl_delay_seconds=crawl_delay,
                )

                response = self._request_with_retry(page_url)
                soup = BeautifulSoup(response.text, "html.parser")
                page_data = self.parse_page(
                    page_kind=page_kind,
                    page_url=page_url,
                    soup=soup,
                )
                parsed.merge(page_data)
                log_event(
                    logger,
                    logging.INFO,
                    "page_scraped",
                    competitor=self.config.name,
                    page_kind=page_kind,
                    page_url=page_url,
                )
            except Exception as exc:
                failed_pages += 1
                message = f"page={page_kind} url={page_url} error={exc}"
                errors.append(message)
                log_event(
                    logger,
                    logging.ERROR,
                    "page_scrape_failed",
                    competitor=self.config.name,
                    page_kind=page_kind,
                    page_url=page_url,
                    error=str(exc),
                )

        normalized = self.normalizer.normalize(
            competitor=self.config,
            parsed=parsed,
            scraped_at=datetime.now(timezone.utc),
        )
        return ScraperRunResult(
            competitor=self.config.name,
            records=normalized,
            failed_pages=failed_pages,
            errors=errors,
        )

    def page_plan(self) -> dict[str, str]:
        """
        Pages to scrape for this competitor.
        """

        return self.config.pages

    @abstractmethod
    def parse_page(
        self,
        *,
        page_kind: str,
        page_url: str,
        soup: BeautifulSoup,
    ) -> ParsedCompetitorData:
        """
        Parse one page and return extracted data.
        """

    def _request_with_retry(self, url: str) -> requests.Response:
        last_error: Exception | None = None

        for attempt in range(self.settings.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    headers=self.request_headers,
                    timeout=self.settings.timeout_seconds,
                    allow_redirects=True,
                )
                if response.status_code in RETRYABLE_STATUS_CODES:
                    raise requests.HTTPError(
                        f"Retryable status={response.status_code}",
                        response=response,
                    )
                response.raise_for_status()
                return response
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                last_error = exc
                if isinstance(exc, requests.HTTPError):
                    status_code = exc.response.status_code if exc.response is not None else None
                    if status_code not in RETRYABLE_STATUS_CODES:
                        raise

            if attempt >= self.settings.max_retries:
                break

            backoff_seconds = self.settings.backoff_initial_seconds * (
                self.settings.backoff_multiplier**attempt
            )
            time.sleep(backoff_seconds)

        raise RuntimeError(f"Failed to fetch {url} after retries: {last_error}")
