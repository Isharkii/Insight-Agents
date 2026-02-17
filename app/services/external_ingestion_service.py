"""
app/services/external_ingestion_service.py

Orchestration service for external source ingestion.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import (
    get_external_http_settings,
    get_external_ingestion_settings,
    get_google_trends_settings,
    get_news_api_settings,
    get_world_bank_settings,
)
from app.connectors import (
    BaseConnector,
    ConnectorRequestError,
    GoogleTrendsConnector,
    NewsAPIConnector,
    WorldBankConnector,
)
from app.domain.external_ingestion import SourceIngestionSummary
from app.repositories.external_ingestion_repository import ExternalIngestionRepository

logger = logging.getLogger(__name__)


class ExternalIngestionService:
    """
    Coordinates connector fetching and database persistence.
    """

    def __init__(
        self,
        *,
        connectors: Sequence[BaseConnector],
        batch_size: int,
    ) -> None:
        self._connectors = {connector.source: connector for connector in connectors}
        self._batch_size = max(1, batch_size)

    def ingest(
        self,
        *,
        db: Session,
        source: str | None = None,
    ) -> list[SourceIngestionSummary]:
        """
        Ingest records from one source or all configured sources.
        """

        selected_connectors = self._select_connectors(source)
        repository = ExternalIngestionRepository(db)
        summaries: list[SourceIngestionSummary] = []

        for connector in selected_connectors:
            try:
                fetched = connector.fetch_records()
            except ConnectorRequestError as exc:
                logger.error(
                    "External connector request failed source=%s error=%s",
                    connector.source,
                    exc,
                )
                summaries.append(
                    SourceIngestionSummary(
                        source=connector.source,
                        records_inserted=0,
                        failed_records=1,
                    )
                )
                continue
            except Exception as exc:
                logger.exception(
                    "Unhandled connector failure source=%s error=%s",
                    connector.source,
                    exc,
                )
                summaries.append(
                    SourceIngestionSummary(
                        source=connector.source,
                        records_inserted=0,
                        failed_records=1,
                    )
                )
                continue

            inserted = 0
            failed = fetched.failed_records
            if fetched.records:
                try:
                    inserted = repository.bulk_insert_records(
                        fetched.records,
                        batch_size=self._batch_size,
                    )
                    db.commit()
                except SQLAlchemyError as exc:
                    db.rollback()
                    logger.exception(
                        "Failed to persist external records source=%s error=%s",
                        connector.source,
                        exc,
                    )
                    failed += len(fetched.records)
                    inserted = 0

            summaries.append(
                SourceIngestionSummary(
                    source=connector.source,
                    records_inserted=inserted,
                    failed_records=failed,
                )
            )

        return summaries

    def _select_connectors(self, source: str | None) -> list[BaseConnector]:
        if source is None:
            return list(self._connectors.values())

        normalized = source.strip().lower()
        connector = self._connectors.get(normalized)
        if connector is None:
            allowed = ", ".join(sorted(self._connectors.keys()))
            raise ValueError(f"Unsupported source '{source}'. Allowed sources: {allowed}.")
        return [connector]


@lru_cache(maxsize=1)
def get_external_ingestion_service() -> ExternalIngestionService:
    """
    Build and cache the external ingestion service.
    """

    http_settings = get_external_http_settings()
    connectors: list[BaseConnector] = [
        NewsAPIConnector(
            settings=get_news_api_settings(),
            http_settings=http_settings,
        ),
        GoogleTrendsConnector(
            settings=get_google_trends_settings(),
            http_settings=http_settings,
        ),
        WorldBankConnector(
            settings=get_world_bank_settings(),
            http_settings=http_settings,
        ),
    ]
    ingestion_settings = get_external_ingestion_settings()
    return ExternalIngestionService(
        connectors=connectors,
        batch_size=ingestion_settings.batch_size,
    )
