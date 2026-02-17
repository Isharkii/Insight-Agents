"""
Scraper subclass exports.
"""

from app.scraping.scrapers.configurable_scraper import ConfigurableCompetitorScraper
from app.scraping.scrapers.saas_scraper import SaaSCompetitorScraper

__all__ = ["ConfigurableCompetitorScraper", "SaaSCompetitorScraper"]
