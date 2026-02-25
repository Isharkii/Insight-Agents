"""
app/services/macro_ingestion_service.py

Macro data ingestion pipeline: fetch, validate, deduplicate, and persist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

from sqlalchemy.orm import Session

from app.config import (
    get_external_http_settings,
    get_fred_settings,
    get_macro_ingestion_settings,
)
from app.connectors.macro import (
    BaseMacroProvider,
    MacroObservation,
    MacroProviderError,
    MacroProviderRegistry,
    build_default_macro_provider_registry,
)
from app.domain.macro_ingestion import MacroIngestionSummary, MacroSeriesRequest
from db.repositories.macro_metrics_repository import MacroMetricsRepository

MacroRepositoryFactory = Callable[[Session], MacroMetricsRepository]

_MONTHLY = "M"
_QUARTERLY = "Q"


@dataclass(frozen=True)
class _MetricProfile:
    expected_frequency: str
    default_provider_metric_by_source: dict[str, str]


_METRIC_PROFILES: dict[str, _MetricProfile] = {
    "gdp": _MetricProfile(
        expected_frequency=_QUARTERLY,
        default_provider_metric_by_source={
            "fred": "GDP",
            "world_bank": "NY.GDP.MKTP.CD",
        },
    ),
    "cpi": _MetricProfile(
        expected_frequency=_MONTHLY,
        default_provider_metric_by_source={
            "fred": "CPIAUCSL",
            "world_bank": "FP.CPI.TOTL",
        },
    ),
    "policy_rate": _MetricProfile(
        expected_frequency=_MONTHLY,
        default_provider_metric_by_source={
            "fred": "FEDFUNDS",
            "world_bank": "FR.INR.RINR",
        },
    ),
}


class MacroIngestionError(RuntimeError):
    """
    Base macro ingestion pipeline failure.
    """


class MacroIngestionValidationError(MacroIngestionError):
    """
    Raised when ingestion configuration is invalid.
    """


class MacroIngestionProviderError(MacroIngestionError):
    """
    Raised when provider fetching fails.
    """


class MacroIngestionService:
    """
    Deterministic ingestion pipeline for macro time-series.
    """

    def __init__(
        self,
        *,
        provider_registry: MacroProviderRegistry,
        default_provider: str = "fred",
        default_country_code: str = "US",
        batch_size: int = 500,
        repository_factory: MacroRepositoryFactory = MacroMetricsRepository,
    ) -> None:
        self._provider_registry = provider_registry
        self._default_provider = str(default_provider or "fred").strip().lower()
        self._default_country_code = str(default_country_code or "US").strip().upper()
        self._batch_size = max(1, int(batch_size))
        self._repository_factory = repository_factory

    def ingest(
        self,
        *,
        db: Session,
        country_code: str | None = None,
        provider_name: str | None = None,
        series_requests: Sequence[MacroSeriesRequest] | None = None,
        run_version: int | None = None,
        source_release_ts: datetime | None = None,
        run_metadata: Mapping[str, Any] | None = None,
        mark_current: bool = True,
    ) -> MacroIngestionSummary:
        """
        Fetch and persist macro metrics with idempotent versioned upserts.
        """

        provider = self._resolve_provider(provider_name)
        normalized_country = self._normalize_country(country_code or self._default_country_code)
        resolved_requests = self._resolve_requests(
            provider_name=provider.source,
            requests=series_requests,
        )
        ingestion_ts = datetime.now(tz=timezone.utc)

        fetched_count = 0
        valid_rows: list[dict[str, Any]] = []
        validation_errors: list[str] = []

        for request in resolved_requests:
            profile = self._profile_for_metric(request.metric_name)
            provider_metric = self._provider_metric_for_request(
                provider_name=provider.source,
                request=request,
                profile=profile,
            )
            try:
                observations = provider.fetch(
                    country=normalized_country,
                    metric=provider_metric,
                    period_start=request.period_start,
                    period_end=request.period_end,
                    limit=request.limit,
                )
            except MacroProviderError as exc:
                raise MacroIngestionProviderError(str(exc)) from exc

            fetched_count += len(observations)
            for observation in observations:
                transformed, error = self._transform_observation(
                    observation=observation,
                    request=request,
                    provider_name=provider.source,
                    provider_metric=provider_metric,
                    country_code=normalized_country,
                    ingestion_ts=ingestion_ts,
                )
                if transformed is None:
                    if error:
                        validation_errors.append(error)
                    continue
                valid_rows.append(transformed)

        deduped_rows = _deduplicate_metric_rows(valid_rows)
        resolved_run_version = int(run_version) if run_version is not None else None

        metadata_json = self._build_run_metadata(
            provider_name=provider.source,
            country_code=normalized_country,
            requests=resolved_requests,
            fetched_count=fetched_count,
            valid_count=len(deduped_rows),
            validation_errors=validation_errors,
            run_metadata=run_metadata,
            ingestion_ts=ingestion_ts,
        )

        with self._transaction_context(db):
            repository = self._repository_factory(db)
            if resolved_run_version is None:
                resolved_run_version = repository.next_run_version(
                    source_key=provider.source,
                    country_code=normalized_country,
                )

            run_id = repository.upsert_run(
                source_key=provider.source,
                country_code=normalized_country,
                run_version=resolved_run_version,
                source_release_ts=source_release_ts,
                is_current=mark_current,
                metadata_json=metadata_json,
            )
            if mark_current:
                repository.mark_only_current(
                    source_key=provider.source,
                    country_code=normalized_country,
                    current_run_id=run_id,
                )

            upserted_count = repository.bulk_upsert_metrics(
                run_id=run_id,
                rows=deduped_rows,
                batch_size=self._batch_size,
            )

        return MacroIngestionSummary(
            source_key=provider.source,
            country_code=normalized_country,
            run_id=run_id,
            run_version=resolved_run_version,
            fetched_records=fetched_count,
            valid_records=len(deduped_rows),
            upserted_records=upserted_count,
            skipped_records=max(0, fetched_count - len(deduped_rows)),
            validation_errors=tuple(validation_errors),
            ingested_at=ingestion_ts,
        )

    def _resolve_provider(self, provider_name: str | None) -> BaseMacroProvider:
        target = str(provider_name or self._default_provider).strip().lower()
        if not target:
            raise MacroIngestionValidationError("Macro provider name cannot be empty.")
        return self._provider_registry.get(target)

    def _resolve_requests(
        self,
        *,
        provider_name: str,
        requests: Sequence[MacroSeriesRequest] | None,
    ) -> list[MacroSeriesRequest]:
        if requests:
            return [request for request in requests]

        defaults: list[MacroSeriesRequest] = []
        for metric_name in ("gdp", "cpi", "policy_rate"):
            profile = self._profile_for_metric(metric_name)
            provider_metric = profile.default_provider_metric_by_source.get(provider_name)
            defaults.append(
                MacroSeriesRequest(
                    metric_name=metric_name,
                    provider_metric=provider_metric,
                )
            )
        return defaults

    def _profile_for_metric(self, metric_name: str) -> _MetricProfile:
        normalized = str(metric_name or "").strip().lower()
        profile = _METRIC_PROFILES.get(normalized)
        if profile is None:
            supported = ", ".join(sorted(_METRIC_PROFILES.keys()))
            raise MacroIngestionValidationError(
                f"Unsupported metric '{metric_name}'. Supported metrics: {supported}."
            )
        return profile

    def _provider_metric_for_request(
        self,
        *,
        provider_name: str,
        request: MacroSeriesRequest,
        profile: _MetricProfile,
    ) -> str:
        provider_metric = str(request.provider_metric or "").strip()
        if provider_metric:
            return provider_metric
        default_value = profile.default_provider_metric_by_source.get(provider_name)
        if default_value:
            return default_value
        raise MacroIngestionValidationError(
            f"Missing provider metric for '{request.metric_name}' on provider '{provider_name}'."
        )

    def _transform_observation(
        self,
        *,
        observation: MacroObservation,
        request: MacroSeriesRequest,
        provider_name: str,
        provider_metric: str,
        country_code: str,
        ingestion_ts: datetime,
    ) -> tuple[dict[str, Any] | None, str | None]:
        profile = self._profile_for_metric(request.metric_name)
        metric_name = str(request.metric_name).strip().lower()
        raw_period_start = str(observation.get("period_start") or "").strip()
        raw_period_end = str(observation.get("period_end") or "").strip()
        parsed_start = _parse_date(raw_period_start)
        parsed_end = _parse_date(raw_period_end)

        if parsed_start is None or parsed_end is None:
            return (
                None,
                f"{metric_name}: invalid period bounds start={raw_period_start!r} end={raw_period_end!r}",
            )
        if parsed_start > parsed_end:
            return (
                None,
                f"{metric_name}: period_start after period_end start={raw_period_start!r} end={raw_period_end!r}",
            )

        if parsed_start == parsed_end:
            parsed_start, parsed_end = _expand_single_day_window(parsed_start, profile.expected_frequency)

        inferred_frequency = _infer_frequency(parsed_start, parsed_end)
        if inferred_frequency != profile.expected_frequency:
            return (
                None,
                f"{metric_name}: expected {profile.expected_frequency} frequency, got {inferred_frequency}.",
            )

        numeric_value = _coerce_float(observation.get("value"))
        if numeric_value is None:
            return (None, f"{metric_name}: non-numeric value={observation.get('value')!r}")

        row_metadata: dict[str, Any] = {
            "provider": provider_name,
            "provider_metric": provider_metric,
            "provider_record_metric": str(observation.get("metric") or provider_metric),
            "provider_country": str(observation.get("country") or country_code).upper(),
            "observation_source": str(observation.get("source") or provider_name),
            "raw_period_start": raw_period_start,
            "raw_period_end": raw_period_end,
            "ingested_at": ingestion_ts.isoformat(),
        }
        if request.metadata:
            row_metadata.update(dict(request.metadata))

        return (
            {
                "country_code": country_code,
                "metric_name": metric_name,
                "frequency": profile.expected_frequency,
                "period_start": parsed_start,
                "period_end": parsed_end,
                "value": numeric_value,
                "unit": request.unit,
                "metadata_json": row_metadata,
            },
            None,
        )

    @staticmethod
    def _normalize_country(country_code: str) -> str:
        normalized = str(country_code or "").strip().upper()
        if len(normalized) < 2 or len(normalized) > 3:
            raise MacroIngestionValidationError(
                f"country_code '{country_code}' is invalid. Expected ISO-2 or ISO-3 code."
            )
        return normalized

    @staticmethod
    def _build_run_metadata(
        *,
        provider_name: str,
        country_code: str,
        requests: Sequence[MacroSeriesRequest],
        fetched_count: int,
        valid_count: int,
        validation_errors: Sequence[str],
        run_metadata: Mapping[str, Any] | None,
        ingestion_ts: datetime,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": provider_name,
            "country_code": country_code,
            "requested_metrics": [str(item.metric_name).strip().lower() for item in requests],
            "fetched_records": int(fetched_count),
            "valid_records": int(valid_count),
            "skipped_records": int(max(0, fetched_count - valid_count)),
            "validation_error_count": len(validation_errors),
            "validation_errors": list(validation_errors[:100]),
            "ingestion_timestamp": ingestion_ts.isoformat(),
        }
        if run_metadata:
            payload["request_metadata"] = dict(run_metadata)
        return payload

    @staticmethod
    def _transaction_context(db: Session) -> Any:
        if db.in_transaction():
            return db.begin_nested()
        return db.begin()


@lru_cache(maxsize=1)
def get_macro_ingestion_service() -> MacroIngestionService:
    """
    Build and cache macro ingestion pipeline service.
    """

    macro_settings = get_macro_ingestion_settings()
    fred_settings = get_fred_settings()

    registry = build_default_macro_provider_registry(
        http_settings=get_external_http_settings(),
        fred_api_key=fred_settings.api_key,
        fred_base_url=fred_settings.base_url,
    )
    return MacroIngestionService(
        provider_registry=registry,
        default_provider=macro_settings.default_provider,
        default_country_code=macro_settings.default_country_code,
        batch_size=macro_settings.batch_size,
    )


def _parse_date(value: str) -> date | None:
    if not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return date.fromisoformat(normalized[:10])
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _expand_single_day_window(value: date, expected_frequency: str) -> tuple[date, date]:
    if expected_frequency == _MONTHLY:
        return _month_bounds(value)
    if expected_frequency == _QUARTERLY:
        return _quarter_bounds(value)
    return value, value


def _month_bounds(value: date) -> tuple[date, date]:
    start = date(year=value.year, month=value.month, day=1)
    if value.month == 12:
        next_month = date(year=value.year + 1, month=1, day=1)
    else:
        next_month = date(year=value.year, month=value.month + 1, day=1)
    end = next_month - timedelta(days=1)
    return start, end


def _quarter_bounds(value: date) -> tuple[date, date]:
    quarter_start_month = ((value.month - 1) // 3) * 3 + 1
    start = date(year=value.year, month=quarter_start_month, day=1)
    if quarter_start_month == 10:
        quarter_end = date(year=value.year, month=12, day=31)
    else:
        next_quarter_month = quarter_start_month + 3
        next_quarter_start = date(year=value.year, month=next_quarter_month, day=1)
        quarter_end = next_quarter_start - timedelta(days=1)
    return start, quarter_end


def _infer_frequency(period_start: date, period_end: date) -> str:
    if period_start > period_end:
        return "U"

    months_span = ((period_end.year - period_start.year) * 12) + (period_end.month - period_start.month) + 1
    month_start, month_end = _month_bounds(period_start)
    quarter_start, quarter_end = _quarter_bounds(period_start)

    if months_span == 1 and period_start == month_start and period_end == month_end:
        return _MONTHLY
    if months_span == 3 and period_start == quarter_start and period_end == quarter_end:
        return _QUARTERLY
    return "U"


def _deduplicate_metric_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str, str, date], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["country_code"]).upper(),
            str(row["metric_name"]).lower(),
            str(row["frequency"]).upper(),
            row["period_end"],
        )
        deduped[key] = dict(row)
    return list(deduped.values())
