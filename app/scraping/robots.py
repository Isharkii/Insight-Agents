"""
robots.txt policy helper for scraper compliance.
"""

from __future__ import annotations

import logging
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests

from app.scraping.logging_utils import log_event

logger = logging.getLogger(__name__)


class RobotsPolicyManager:
    """
    Caches robots.txt rules and access checks per origin.
    """

    def __init__(
        self,
        *,
        session: requests.Session,
        timeout_seconds: float = 10.0,
        allow_when_unreachable: bool = True,
    ) -> None:
        self._session = session
        self._timeout_seconds = timeout_seconds
        self._allow_when_unreachable = allow_when_unreachable
        self._cache: dict[str, RobotFileParser] = {}

    def can_fetch(self, *, url: str, user_agent: str) -> bool:
        """
        Return whether scraping `url` is allowed for `user_agent`.
        """

        parser = self._get_parser(url)
        return parser.can_fetch(user_agent, url)

    def crawl_delay(self, *, url: str, user_agent: str) -> float | None:
        """
        Return crawl-delay if published in robots.txt for this URL.
        """

        parser = self._get_parser(url)
        delay = parser.crawl_delay(user_agent)
        if delay is None:
            delay = parser.crawl_delay("*")
        return delay

    def _get_parser(self, url: str) -> RobotFileParser:
        origin = self._origin(url)
        cached = self._cache.get(origin)
        if cached is not None:
            return cached

        parser = RobotFileParser()
        robots_url = urljoin(origin, "/robots.txt")
        try:
            response = self._session.get(
                robots_url,
                timeout=self._timeout_seconds,
            )
            if response.ok and response.text:
                parser.set_url(robots_url)
                parser.parse(response.text.splitlines())
                log_event(
                    logger,
                    logging.INFO,
                    "robots_loaded",
                    origin=origin,
                    robots_url=robots_url,
                )
            else:
                self._apply_fallback_policy(parser)
                log_event(
                    logger,
                    logging.WARNING,
                    "robots_unavailable",
                    origin=origin,
                    robots_url=robots_url,
                    status_code=response.status_code,
                    fallback_allow=self._allow_when_unreachable,
                )
        except requests.RequestException as exc:
            self._apply_fallback_policy(parser)
            log_event(
                logger,
                logging.WARNING,
                "robots_fetch_failed",
                origin=origin,
                robots_url=robots_url,
                fallback_allow=self._allow_when_unreachable,
                error=str(exc),
            )

        self._cache[origin] = parser
        return parser

    def _apply_fallback_policy(self, parser: RobotFileParser) -> None:
        if self._allow_when_unreachable:
            parser.parse(["User-agent: *", "Allow: /"])
        else:
            parser.parse(["User-agent: *", "Disallow: /"])

    @staticmethod
    def _origin(url: str) -> str:
        parsed = urlparse(url)
        scheme = parsed.scheme or "https"
        return f"{scheme}://{parsed.netloc}"
