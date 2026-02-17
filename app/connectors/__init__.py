"""
app/connectors package marker.
"""

from app.connectors.base import BaseConnector, ConnectorFetchResult, ConnectorRequestError
from app.connectors.google_trends_connector import GoogleTrendsConnector
from app.connectors.news_api_connector import NewsAPIConnector
from app.connectors.world_bank_connector import WorldBankConnector

__all__ = [
    "BaseConnector",
    "ConnectorFetchResult",
    "ConnectorRequestError",
    "NewsAPIConnector",
    "GoogleTrendsConnector",
    "WorldBankConnector",
]
