# Macro Ingestion Pipeline

The macro ingestion pipeline is implemented in:

- `app/services/macro_ingestion_service.py`
- `db/repositories/macro_metrics_repository.py`

## Flow

1. Resolve provider from `MacroProviderRegistry` (`fred`, `world_bank`, `imf` placeholder).
2. Fetch normalized observations from provider interface:
   - `country`
   - `metric`
   - `period_start`
   - `period_end`
   - `value`
   - `source`
3. Validate each observation against canonical metric rules:
   - `gdp` -> quarterly (`Q`)
   - `cpi` -> monthly (`M`)
   - `policy_rate` -> monthly (`M`)
4. Normalize single-day observations into full monthly/quarterly windows.
5. Deduplicate rows in memory (last write wins by `country + metric + frequency + period_end`).
6. Persist in one transaction:
   - Upsert run header in `macro_metric_runs` with `run_version` and metadata.
   - Mark only one run as current for `(source_key, country_code)`.
   - Bulk upsert facts in `macro_metrics`.

## Idempotency and Upserts

- Re-running the same `(source_key, country_code, run_version)` updates the same run row.
- Metric rows upsert on:
  - `run_id`
  - `country_code`
  - `metric_name`
  - `frequency`
  - `period_end`
- Duplicate records in one ingestion call are collapsed before DB writes.

## Metadata and Versioning

- Run-level metadata is stored in `macro_metric_runs.metadata_json`.
- Record-level provenance is stored in `macro_metrics.metadata_json`.
- `run_version` supports historical snapshots and reproducibility.

## Transaction Handling

- Uses `Session.begin()` when no transaction exists.
- Uses `Session.begin_nested()` when called inside an existing transaction.
- Failures roll back the active transaction scope automatically.

