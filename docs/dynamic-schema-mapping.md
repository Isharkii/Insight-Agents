# Dynamic Schema Mapping Engine

## Purpose

Map arbitrary client CSV schemas into canonical records for `canonical_insight_records`.

## Architecture

- `SchemaMapper`: `app/mappers/schema_mapper.py`
- `MappingValidator`: `app/validators/mapping_validator.py`
- `MappingConfig` model: `db/models/mapping_config.py`
- `MappingConfigRepository`: `app/repositories/mapping_config_repository.py`

## Behavior

- Auto-detects canonical mappings via exact, alias, and fuzzy matching.
- Applies manual overrides from:
  - DB mapping config (`field_mapping_json`)
  - runtime overrides (if passed programmatically)
- Enforces required canonical mappings and raises structured `SchemaMappingError`.
- Produces mapped `CanonicalInsightRecord` objects via `SchemaMapper.to_canonical_record(...)`.
