from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import pytest

from app.config import ExternalHTTPSettings
from app.connectors.macro import BaseMacroProvider, MacroObservation, MacroProviderRegistry
from app.domain.macro_ingestion import MacroSeriesRequest
from app.services.macro_ingestion_service import MacroIngestionService


class _StaticMacroProvider(BaseMacroProvider):
    provider_name = "test_provider"

    def __init__(self, observations_by_metric: dict[str, list[MacroObservation]]) -> None:
        super().__init__(
            source=self.provider_name,
            http_settings=ExternalHTTPSettings(rate_limit_per_second=0.0),
        )
        self._observations_by_metric = observations_by_metric

    def fetch(
        self,
        *,
        country: str,
        metric: str,
        period_start: str | None = None,
        period_end: str | None = None,
        limit: int | None = None,
    ) -> list[MacroObservation]:
        _ = (country, period_start, period_end, limit)
        return [dict(item) for item in self._observations_by_metric.get(metric, [])]


class _FakeTransaction:
    def __init__(self, owner: "_FakeSession", *, nested: bool) -> None:
        self._owner = owner
        self._nested = nested

    def __enter__(self) -> "_FakeTransaction":
        self._owner._depth += 1
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
        _ = (exc, tb, self._nested)
        self._owner._depth = max(0, self._owner._depth - 1)
        if exc_type is None:
            self._owner.commits += 1
        else:
            self._owner.rollbacks += 1
        return False


class _FakeSession:
    def __init__(self) -> None:
        self._depth = 0
        self.begin_calls = 0
        self.begin_nested_calls = 0
        self.commits = 0
        self.rollbacks = 0

    def in_transaction(self) -> bool:
        return self._depth > 0

    def begin(self) -> _FakeTransaction:
        self.begin_calls += 1
        return _FakeTransaction(self, nested=False)

    def begin_nested(self) -> _FakeTransaction:
        self.begin_nested_calls += 1
        return _FakeTransaction(self, nested=True)


class _FakeRepository:
    latest_version = 4

    def __init__(self, session: _FakeSession) -> None:
        self.session = session
        self.run_payloads: list[dict[str, Any]] = []
        self.mark_current_payloads: list[dict[str, Any]] = []
        self.metric_rows: list[dict[str, Any]] = []
        self.raise_on_bulk = False
        self.run_id = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

    def next_run_version(self, *, source_key: str, country_code: str) -> int:
        _ = (source_key, country_code)
        return self.latest_version + 1

    def upsert_run(
        self,
        *,
        source_key: str,
        country_code: str,
        run_version: int,
        source_release_ts: datetime | None,
        is_current: bool,
        metadata_json: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        self.run_payloads.append(
            {
                "source_key": source_key,
                "country_code": country_code,
                "run_version": run_version,
                "source_release_ts": source_release_ts,
                "is_current": is_current,
                "metadata_json": metadata_json or {},
            }
        )
        return self.run_id

    def mark_only_current(self, *, source_key: str, country_code: str, current_run_id: uuid.UUID) -> int:
        self.mark_current_payloads.append(
            {
                "source_key": source_key,
                "country_code": country_code,
                "current_run_id": current_run_id,
            }
        )
        return 1

    def bulk_upsert_metrics(
        self,
        *,
        run_id: uuid.UUID,
        rows: list[dict[str, Any]],
        batch_size: int = 500,
    ) -> int:
        _ = batch_size
        if self.raise_on_bulk:
            raise RuntimeError("forced bulk failure")
        for row in rows:
            copied = dict(row)
            copied["run_id"] = run_id
            self.metric_rows.append(copied)
        return len(rows)


def _build_service(
    provider: BaseMacroProvider,
    repository: _FakeRepository,
) -> MacroIngestionService:
    registry = MacroProviderRegistry({"test_provider": provider})
    return MacroIngestionService(
        provider_registry=registry,
        default_provider="test_provider",
        default_country_code="US",
        repository_factory=lambda _: repository,
    )


def test_ingestion_pipeline_deduplicates_and_upserts() -> None:
    provider = _StaticMacroProvider(
        {
            "GDP": [
                {
                    "country": "US",
                    "metric": "GDP",
                    "period_start": "2025-04-01",
                    "period_end": "2025-04-01",
                    "value": 28000.5,
                    "source": "test_provider",
                }
            ],
            "CPIAUCSL": [
                {
                    "country": "US",
                    "metric": "CPIAUCSL",
                    "period_start": "2025-01-01",
                    "period_end": "2025-01-01",
                    "value": 300.0,
                    "source": "test_provider",
                },
                {
                    "country": "US",
                    "metric": "CPIAUCSL",
                    "period_start": "2025-01-01",
                    "period_end": "2025-01-01",
                    "value": 301.0,
                    "source": "test_provider",
                },
            ],
            "FEDFUNDS": [
                {
                    "country": "US",
                    "metric": "FEDFUNDS",
                    "period_start": "2025-01-01",
                    "period_end": "2025-01-01",
                    "value": 5.25,
                    "source": "test_provider",
                }
            ],
        }
    )
    session = _FakeSession()
    repository = _FakeRepository(session)
    service = _build_service(provider, repository)

    summary = service.ingest(
        db=session,  # type: ignore[arg-type]
        country_code="US",
        provider_name="test_provider",
        run_version=7,
        series_requests=[
            MacroSeriesRequest(metric_name="gdp", provider_metric="GDP"),
            MacroSeriesRequest(metric_name="cpi", provider_metric="CPIAUCSL"),
            MacroSeriesRequest(metric_name="policy_rate", provider_metric="FEDFUNDS"),
        ],
    )

    assert summary.run_version == 7
    assert summary.fetched_records == 4
    assert summary.valid_records == 3
    assert summary.upserted_records == 3
    assert summary.skipped_records == 1
    assert summary.validation_errors == ()

    assert session.begin_calls == 1
    assert session.commits == 1
    assert session.rollbacks == 0
    assert len(repository.mark_current_payloads) == 1
    assert len(repository.metric_rows) == 3

    by_metric = {row["metric_name"]: row for row in repository.metric_rows}
    assert by_metric["gdp"]["frequency"] == "Q"
    assert by_metric["cpi"]["frequency"] == "M"
    assert by_metric["policy_rate"]["frequency"] == "M"
    assert by_metric["cpi"]["value"] == 301.0


def test_ingestion_pipeline_skips_invalid_frequency_records() -> None:
    provider = _StaticMacroProvider(
        {
            "GDP": [
                {
                    "country": "US",
                    "metric": "GDP",
                    "period_start": "2025-02-10",
                    "period_end": "2025-02-20",
                    "value": 28000.5,
                    "source": "test_provider",
                }
            ],
        }
    )
    session = _FakeSession()
    repository = _FakeRepository(session)
    service = _build_service(provider, repository)

    summary = service.ingest(
        db=session,  # type: ignore[arg-type]
        series_requests=[MacroSeriesRequest(metric_name="gdp", provider_metric="GDP")],
    )

    assert summary.run_version == 5
    assert summary.fetched_records == 1
    assert summary.valid_records == 0
    assert summary.upserted_records == 0
    assert summary.skipped_records == 1
    assert len(summary.validation_errors) == 1
    assert "expected Q frequency" in summary.validation_errors[0]
    assert repository.metric_rows == []


def test_ingestion_pipeline_uses_nested_transaction_and_rolls_back() -> None:
    provider = _StaticMacroProvider(
        {
            "GDP": [
                {
                    "country": "US",
                    "metric": "GDP",
                    "period_start": "2025-01-01",
                    "period_end": "2025-01-01",
                    "value": 28000.5,
                    "source": "test_provider",
                }
            ]
        }
    )
    session = _FakeSession()
    session._depth = 1  # Simulate existing outer transaction.
    repository = _FakeRepository(session)
    repository.raise_on_bulk = True
    service = _build_service(provider, repository)

    with pytest.raises(RuntimeError, match="forced bulk failure"):
        service.ingest(
            db=session,  # type: ignore[arg-type]
            series_requests=[MacroSeriesRequest(metric_name="gdp", provider_metric="GDP")],
        )

    assert session.begin_nested_calls == 1
    assert session.commits == 0
    assert session.rollbacks == 1
