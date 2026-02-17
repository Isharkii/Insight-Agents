"""
Config helpers for competitor scraping.
"""

from app.scraping.config.loader import get_competitor_scraping_settings, load_competitor_configs
from app.scraping.config.models import CompetitorDomainConfig, CompetitorScrapingSettings

__all__ = [
    "CompetitorDomainConfig",
    "CompetitorScrapingSettings",
    "get_competitor_scraping_settings",
    "load_competitor_configs",
]
