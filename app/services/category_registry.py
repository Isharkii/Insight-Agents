"""
app/services/category_registry.py

YAML-backed category registry for KPI orchestration.

Each category pack defines:
- metric aliases
- required inputs
- deterministic formula class and input bindings
- optional signals
"""

from __future__ import annotations

import copy
import importlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from kpi.base import BaseKPIFormula

_DEFAULT_CATEGORIES_DIR = Path(__file__).resolve().parents[2] / "config" / "categories"
_SUPPORTED_MISSING_WHEN = {"is_none", "is_empty"}
_DEFAULT_PROCESSING_STRATEGY = "general_timeseries"
_DEFAULT_PRIMARY_METRIC = "revenue"
_PRIMARY_METRIC_ALIAS_KEY = "recurring_revenue"


class CategoryRegistryError(RuntimeError):
    """Raised when category packs cannot be loaded or validated."""


@dataclass(frozen=True)
class FormulaInputBinding:
    """Maps one formula input key to an aggregated/external source."""

    source: str
    default: Any


@dataclass(frozen=True)
class DependencyRule:
    """Describes when one dependency is considered missing."""

    source: str
    missing_when: str = "is_none"


@dataclass(frozen=True)
class CategoryPack:
    """Fully-resolved category definition loaded from YAML."""

    name: str
    metric_aliases: dict[str, tuple[str, ...]]
    required_inputs: tuple[str, ...]
    deterministic_formula_class: str
    formula: BaseKPIFormula
    formula_input_bindings: dict[str, FormulaInputBinding]
    validity_rules: dict[str, tuple[DependencyRule, ...]]
    optional_signals: tuple[str, ...]
    category_aliases: tuple[str, ...]
    rate_metrics: frozenset[str]


def _normalize(value: str | None) -> str:
    return str(value or "").strip().lower()


def _categories_dir() -> Path:
    raw = os.getenv("CATEGORY_CONFIG_DIR", "").strip()
    if raw:
        return Path(raw)
    return _DEFAULT_CATEGORIES_DIR


def _load_pack_document(path: Path) -> dict[str, Any]:
    raw_text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]

        payload = yaml.safe_load(raw_text)
    except ModuleNotFoundError:
        # JSON is valid YAML 1.2. This keeps runtime light when packs use
        # JSON-compatible YAML syntax.
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise CategoryRegistryError(
                f"Failed to parse category pack {path.name!r}. "
                "Install PyYAML or use JSON-compatible YAML."
            ) from exc

    if not isinstance(payload, dict):
        raise CategoryRegistryError(f"Category pack {path.name!r} must be a mapping at top level.")
    return payload


def _unique_normalized(values: list[str], *, fallback: str | None = None) -> tuple[str, ...]:
    out: list[str] = []
    for raw in values:
        val = _normalize(raw)
        if val and val not in out:
            out.append(val)
    if not out and fallback:
        return (fallback,)
    return tuple(out)


def _parse_metric_aliases(raw: Any, *, pack_name: str) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, dict) or not raw:
        raise CategoryRegistryError(f"{pack_name}: metric_aliases must be a non-empty mapping.")

    parsed: dict[str, tuple[str, ...]] = {}
    for raw_key, raw_aliases in raw.items():
        key = _normalize(str(raw_key))
        if not key:
            raise CategoryRegistryError(f"{pack_name}: metric_aliases contains an empty key.")

        aliases_source: list[str]
        if isinstance(raw_aliases, str):
            aliases_source = [raw_aliases]
        elif isinstance(raw_aliases, list):
            aliases_source = [str(item) for item in raw_aliases]
        else:
            raise CategoryRegistryError(
                f"{pack_name}: metric_aliases[{key!r}] must be a string or list of strings."
            )

        aliases = _unique_normalized([key, *aliases_source], fallback=key)
        parsed[key] = aliases
    return parsed


def _parse_required_inputs(raw: Any, *, pack_name: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise CategoryRegistryError(f"{pack_name}: required_inputs must be a non-empty list.")
    values = _unique_normalized([str(item) for item in raw])
    if not values:
        raise CategoryRegistryError(f"{pack_name}: required_inputs cannot be empty after normalization.")
    return values


def _parse_formula_class(class_path: str, *, pack_name: str) -> BaseKPIFormula:
    module_name, sep, class_name = class_path.partition(":")
    if not sep or not module_name.strip() or not class_name.strip():
        raise CategoryRegistryError(
            f"{pack_name}: deterministic_formulas.class must use 'module:ClassName' syntax."
        )

    module = importlib.import_module(module_name.strip())
    formula_cls = getattr(module, class_name.strip(), None)
    if formula_cls is None:
        raise CategoryRegistryError(
            f"{pack_name}: formula class {class_name.strip()!r} not found in {module_name.strip()!r}."
        )

    formula = formula_cls()
    if not isinstance(formula, BaseKPIFormula):
        raise CategoryRegistryError(
            f"{pack_name}: {class_path!r} is not a BaseKPIFormula implementation."
        )
    return formula


def _parse_formula_input_bindings(raw: Any, *, pack_name: str) -> dict[str, FormulaInputBinding]:
    if not isinstance(raw, dict) or not raw:
        raise CategoryRegistryError(
            f"{pack_name}: deterministic_formulas.input_bindings must be a non-empty mapping."
        )

    bindings: dict[str, FormulaInputBinding] = {}
    for raw_key, entry in raw.items():
        key = str(raw_key).strip()
        if not key:
            raise CategoryRegistryError(f"{pack_name}: formula input key cannot be empty.")
        if not isinstance(entry, dict):
            raise CategoryRegistryError(
                f"{pack_name}: input_bindings[{key!r}] must be a mapping with source/default."
            )

        source = str(entry.get("source", "")).strip()
        if not source or "." not in source:
            raise CategoryRegistryError(
                f"{pack_name}: input_bindings[{key!r}] has invalid source {source!r}."
            )
        source_prefix = source.split(".", 1)[0].strip().lower()
        if source_prefix not in {"agg", "extra"}:
            raise CategoryRegistryError(
                f"{pack_name}: source prefix must be 'agg' or 'extra', got {source!r}."
            )

        default = copy.deepcopy(entry.get("default"))
        bindings[key] = FormulaInputBinding(source=source, default=default)
    return bindings


def _parse_validity_rules(raw: Any, *, pack_name: str) -> dict[str, tuple[DependencyRule, ...]]:
    if raw in (None, {}):
        return {}
    if not isinstance(raw, dict):
        raise CategoryRegistryError(f"{pack_name}: validity_rules must be a mapping.")

    parsed: dict[str, tuple[DependencyRule, ...]] = {}
    for raw_metric, entry in raw.items():
        metric = _normalize(str(raw_metric))
        if not metric:
            raise CategoryRegistryError(f"{pack_name}: validity_rules has empty metric key.")
        if not isinstance(entry, dict):
            raise CategoryRegistryError(
                f"{pack_name}: validity_rules[{metric!r}] must be a mapping."
            )

        deps_raw = entry.get("dependencies", [])
        if not isinstance(deps_raw, list):
            raise CategoryRegistryError(
                f"{pack_name}: validity_rules[{metric!r}].dependencies must be a list."
            )

        deps: list[DependencyRule] = []
        for dep in deps_raw:
            if isinstance(dep, str):
                source = dep.strip()
                missing_when = "is_none"
            elif isinstance(dep, dict):
                source = str(dep.get("source", "")).strip()
                missing_when = _normalize(str(dep.get("missing_when", "is_none")))
            else:
                raise CategoryRegistryError(
                    f"{pack_name}: dependency for metric {metric!r} must be string or mapping."
                )

            if not source or "." not in source:
                raise CategoryRegistryError(
                    f"{pack_name}: invalid dependency source {source!r} for metric {metric!r}."
                )
            if missing_when not in _SUPPORTED_MISSING_WHEN:
                raise CategoryRegistryError(
                    f"{pack_name}: unsupported missing_when {missing_when!r} for metric {metric!r}."
                )
            deps.append(DependencyRule(source=source, missing_when=missing_when))

        parsed[metric] = tuple(deps)

    return parsed


def _parse_category_pack(path: Path) -> CategoryPack:
    payload = _load_pack_document(path)
    name = _normalize(str(payload.get("name", path.stem)))
    if not name:
        raise CategoryRegistryError(f"{path.name}: category name cannot be empty.")

    metric_aliases = _parse_metric_aliases(payload.get("metric_aliases"), pack_name=name)
    required_inputs = _parse_required_inputs(payload.get("required_inputs"), pack_name=name)
    missing_aliases = [key for key in required_inputs if key not in metric_aliases]
    if missing_aliases:
        raise CategoryRegistryError(
            f"{name}: required_inputs keys missing in metric_aliases: {missing_aliases}"
        )

    formula_block = payload.get("deterministic_formulas")
    if not isinstance(formula_block, dict):
        raise CategoryRegistryError(f"{name}: deterministic_formulas must be a mapping.")
    formula_class = str(formula_block.get("class", "")).strip()
    if not formula_class:
        raise CategoryRegistryError(f"{name}: deterministic_formulas.class is required.")
    formula = _parse_formula_class(formula_class, pack_name=name)
    formula_input_bindings = _parse_formula_input_bindings(
        formula_block.get("input_bindings"),
        pack_name=name,
    )
    validity_rules = _parse_validity_rules(
        formula_block.get("validity_rules"),
        pack_name=name,
    )

    if "optional_signals" not in payload:
        raise CategoryRegistryError(f"{name}: optional_signals is required.")
    optional_signals_raw = payload.get("optional_signals")
    if not isinstance(optional_signals_raw, list):
        raise CategoryRegistryError(f"{name}: optional_signals must be a list.")
    optional_signals = _unique_normalized([str(item) for item in optional_signals_raw])
    category_aliases = _unique_normalized(
        [str(item) for item in payload.get("category_aliases", [name])],
        fallback=name,
    )
    rate_metrics = frozenset(
        _unique_normalized([str(item) for item in payload.get("rate_metrics", [])])
    )

    return CategoryPack(
        name=name,
        metric_aliases=metric_aliases,
        required_inputs=required_inputs,
        deterministic_formula_class=formula_class,
        formula=formula,
        formula_input_bindings=formula_input_bindings,
        validity_rules=validity_rules,
        optional_signals=optional_signals,
        category_aliases=category_aliases,
        rate_metrics=rate_metrics,
    )


def _load_registry() -> dict[str, CategoryPack]:
    categories_dir = _categories_dir()
    if not categories_dir.exists() or not categories_dir.is_dir():
        raise CategoryRegistryError(f"Category config directory not found: {categories_dir}")

    pack_paths = sorted(list(categories_dir.glob("*.yaml")) + list(categories_dir.glob("*.yml")))
    if not pack_paths:
        raise CategoryRegistryError(f"No category packs found in: {categories_dir}")

    registry: dict[str, CategoryPack] = {}
    for pack_path in pack_paths:
        pack = _parse_category_pack(pack_path)
        if pack.name in registry:
            raise CategoryRegistryError(
                f"Duplicate category pack name {pack.name!r} in {pack_path.name!r}."
            )
        registry[pack.name] = pack
    return registry


@lru_cache(maxsize=1)
def _cached_registry() -> dict[str, CategoryPack]:
    return _load_registry()


def clear_category_registry_cache() -> None:
    """Clear cached pack data. Useful for tests and dynamic reloading."""

    _cached_registry.cache_clear()


def supported_categories() -> tuple[str, ...]:
    """Return all configured category names in sorted order."""

    return tuple(sorted(_cached_registry()))


def get_category_pack(category: str | None) -> CategoryPack | None:
    """Return a category pack by name, or None if unknown."""

    return _cached_registry().get(_normalize(category))


def require_category_pack(category: str | None) -> CategoryPack:
    """Return a category pack by name, or raise a descriptive error."""

    pack = get_category_pack(category)
    if pack is None:
        raise CategoryRegistryError(
            f"Unknown category {category!r}. Supported categories: {sorted(_cached_registry())}"
        )
    return pack


def get_processing_strategy(category: str | None) -> str | None:
    """
    Resolve category into an analytics processing strategy.

    Known categories dispatch to their own pack name; unknown non-empty
    categories fall back to ``general_timeseries``.
    """
    normalized = _normalize(category)
    if not normalized:
        return None
    alias = "general_timeseries" if normalized == "generic_timeseries" else normalized
    pack = get_category_pack(alias)
    if pack is None:
        return _DEFAULT_PROCESSING_STRATEGY
    return pack.name


def primary_metric_for_business_type(
    business_type: str | None,
    *,
    fallback: str = _DEFAULT_PRIMARY_METRIC,
) -> str:
    """
    Return the preferred KPI metric for forecast/risk seed selection.

    The value is derived from ``metric_aliases.recurring_revenue`` in the
    category pack. If no non-canonical alias is present, the canonical key
    itself is used.
    """
    pack = get_category_pack(business_type)
    if pack is None:
        return fallback

    aliases = pack.metric_aliases.get(_PRIMARY_METRIC_ALIAS_KEY)
    if not aliases:
        return fallback

    for alias in aliases:
        if alias != _PRIMARY_METRIC_ALIAS_KEY:
            return alias
    return aliases[0]


def churn_metric_for_business_type(
    business_type: str | None,
    *,
    fallback: str = "churn_rate",
) -> str:
    """
    Resolve the churn-like metric name for a business type.

    Prefers any rate metric containing ``"churn"`` and falls back to
    ``"churn_rate"``.
    """
    pack = get_category_pack(business_type)
    if pack is None:
        return fallback

    for metric_name in sorted(pack.rate_metrics):
        if "churn" in metric_name:
            return metric_name
    return fallback
