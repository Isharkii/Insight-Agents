"""
Storage layer exports.
"""

from app.scraping.storage.base import InsightStorage
from app.scraping.storage.sqlalchemy_storage import SQLAlchemyInsightStorage

__all__ = ["InsightStorage", "SQLAlchemyInsightStorage"]
