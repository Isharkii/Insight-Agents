"""Competitive context node for external competitor intelligence integration.

This node is intentionally conservative:
1) It is disabled by default to preserve deterministic behavior.
2) It only emits structured numeric signals into ``state["competitive_context"]``.
3) It never forwards raw scraped web text into state.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from agent.state import AgentState, CompetitiveContext, CompetitiveContextMetric
from app.competitor_intelligence import (
    CompetitorIntelligenceConfig,
    CompetitorIntelligenceService,
    build_competitor_intelligence_service,
)
from app.competitor_intelligence.cache import AsyncTTLCache
from app.competitor_intelligence.schemas import CompetitorIntelligenceRequest

_DEFAULT_COMP_INTEL_TTL_SECONDS = 24 * 60 * 60
_COMP_INTEL_CACHE_MAX_SIZE = 512

_logger = logging.getLogger(__name__)
_SERVICE_LOCK = threading.Lock()
_SERVICE_SINGLETON: CompetitorIntelligenceService | None = None
_COMP_CONTEXT_CACHE: AsyncTTLCache[dict[str, Any]] = AsyncTTLCache(
    ttl_seconds=_DEFAULT_COMP_INTEL_TTL_SECONDS,
    max_size=_COMP_INTEL_CACHE_MAX_SIZE,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_async(awaitable: Any) -> Any:
    """Run async work from sync graph node execution safely."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result: dict[str, Any] = {}
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(awaitable)
        except BaseException as exc:  # pragma: no cover - defensive
            error.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result.get("value")


def _normalize_peer_names(values: Iterable[str], *, entity_name: str) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    entity_norm = str(entity_name or "").strip().lower()
    for raw in values:
        name = str(raw or "").strip()
        if not name:
            continue
        lowered = name.lower()
        if lowered == entity_norm:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        output.append(name)
    return output


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:  # NaN guard
        return None
    return float(numeric)


def _normalize_existing_context(state: AgentState) -> CompetitiveContext:
    raw = state.get("competitive_context")
    if not isinstance(raw, dict):
        return {
            "available": False,
            "source": "unavailable",
            "peer_count": 0,
            "peers": [],
            "metrics": [],
            "benchmark_rows_count": 0,
            "numeric_signals": [],
            "cache_hit": False,
            "generated_at": _now_iso(),
            "warnings": [],
        }

    raw_peers = raw.get("peers", [])
    if not isinstance(raw_peers, list):
        raw_peers = []
    peers = _normalize_peer_names(raw_peers, entity_name=str(state.get("entity_name") or ""))

    raw_metrics = raw.get("metrics", [])
    if not isinstance(raw_metrics, list):
        raw_metrics = []
    metrics = [str(item).strip() for item in raw_metrics if str(item).strip()]
    benchmark_rows = 0
    try:
        benchmark_rows = int(raw.get("benchmark_rows_count", 0))
    except (TypeError, ValueError):
        benchmark_rows = 0
    available = bool(raw.get("available", False))
    source = str(raw.get("source") or ("deterministic_local" if available else "unavailable")).strip() or "unavailable"

    numeric_signals = raw.get("numeric_signals")
    if not isinstance(numeric_signals, list):
        numeric_signals = []
    else:
        numeric_signals = [item for item in numeric_signals if isinstance(item, dict)]

    raw_warnings = raw.get("warnings", [])
    if not isinstance(raw_warnings, list):
        raw_warnings = []

    return {
        "available": available,
        "source": source,
        "peer_count": len(peers),
        "peers": peers,
        "metrics": sorted(set(metrics)),
        "benchmark_rows_count": max(0, benchmark_rows),
        "numeric_signals": numeric_signals,
        "cache_hit": bool(raw.get("cache_hit", False)),
        "generated_at": str(raw.get("generated_at") or _now_iso()),
        "warnings": [str(item) for item in raw_warnings if str(item).strip()],
    }


def _cache_key(*, business_type: str, entity_name: str, peers: list[str]) -> str:
    peer_part = "|".join(sorted(name.lower() for name in peers))
    return f"competitive_context:v1:{business_type.strip().lower()}:{entity_name.strip().lower()}:{peer_part}"


def _build_service_singleton() -> CompetitorIntelligenceService:
    global _SERVICE_SINGLETON
    if _SERVICE_SINGLETON is not None:
        return _SERVICE_SINGLETON
    with _SERVICE_LOCK:
        if _SERVICE_SINGLETON is not None:
            return _SERVICE_SINGLETON
        config = CompetitorIntelligenceConfig.from_env()
        config = config.model_copy(
            update={
                "cache": config.cache.model_copy(
                    update={"ttl_seconds": _DEFAULT_COMP_INTEL_TTL_SECONDS}
                )
            }
        )
        _SERVICE_SINGLETON = build_competitor_intelligence_service(config=config)
        return _SERVICE_SINGLETON


def _context_from_external(
    *,
    state: AgentState,
    existing: CompetitiveContext,
    response: Any,
    cache_hit: bool,
) -> CompetitiveContext:
    metrics: list[CompetitiveContextMetric] = []
    metric_names: list[str] = []
    for item in getattr(response, "aggregated_market_data", []):
        metric_name = str(getattr(item, "metric_name", "")).strip()
        if not metric_name:
            continue
        metric_names.append(metric_name)
        metrics.append(
            {
                "metric_name": metric_name,
                "unit": str(getattr(item, "unit", "ratio") or "ratio"),
                "sample_size": int(getattr(item, "sample_size", 0) or 0),
                "mean": _safe_float(getattr(item, "mean", None)),
                "median": _safe_float(getattr(item, "median", None)),
                "min_value": _safe_float(getattr(item, "min_value", None)),
                "max_value": _safe_float(getattr(item, "max_value", None)),
                "stdev": _safe_float(getattr(item, "stdev", None)),
            }
        )

    external_peers = [
        str(getattr(profile, "competitor_name", "")).strip()
        for profile in getattr(response, "competitor_profiles", [])
        if str(getattr(profile, "competitor_name", "")).strip()
    ]
    merged_peers = _normalize_peer_names(
        [*existing.get("peers", []), *external_peers],
        entity_name=str(state.get("entity_name") or ""),
    )
    warnings = [str(item) for item in getattr(response, "warnings", []) if str(item).strip()]

    available = bool(metrics)
    return {
        "available": available,
        "source": "external_fetch",
        "peer_count": len(merged_peers),
        "peers": merged_peers,
        "metrics": sorted(set(metric_names)),
        "benchmark_rows_count": int(existing.get("benchmark_rows_count", 0) or 0),
        "numeric_signals": metrics,
        "cache_hit": cache_hit,
        "generated_at": _now_iso(),
        "warnings": warnings,
    }


def competitor_node(state: AgentState) -> AgentState:
    """Populate ``state["competitive_context"]`` using numeric-only intelligence."""
    existing_context = _normalize_existing_context(state)
    enabled = _env_bool("COMP_INTEL_ANALYZE_ENABLED", False)
    if not enabled:
        return {
            **state,
            "competitive_context": {
                **existing_context,
                "source": "deterministic_local" if existing_context.get("available") else "disabled",
                "cache_hit": False,
                "generated_at": _now_iso(),
            },
        }

    entity_name = str(state.get("entity_name") or "").strip()
    if not entity_name:
        return {
            **state,
            "competitive_context": {
                **existing_context,
                "available": False,
                "source": "unavailable",
                "warnings": [*existing_context.get("warnings", []), "missing_entity_name"],
                "generated_at": _now_iso(),
            },
        }

    peers = _normalize_peer_names(existing_context.get("peers", []), entity_name=entity_name)
    if not peers:
        return {
            **state,
            "competitive_context": {
                **existing_context,
                "source": "deterministic_local" if existing_context.get("available") else "unavailable",
                "generated_at": _now_iso(),
            },
        }

    key = _cache_key(
        business_type=str(state.get("business_type") or ""),
        entity_name=entity_name,
        peers=peers,
    )
    cached = _run_async(_COMP_CONTEXT_CACHE.get(key))
    if isinstance(cached, dict):
        return {
            **state,
            "competitive_context": {
                **cached,
                "cache_hit": True,
            },
        }

    try:
        service = _build_service_singleton()
        request = CompetitorIntelligenceRequest(
            subject_entity=entity_name,
            competitors=peers,
            market=str(os.getenv("COMP_INTEL_ANALYZE_MARKET", "US") or "US"),
            language=str(os.getenv("COMP_INTEL_ANALYZE_LANGUAGE", "en") or "en"),
            recency_days=_env_int("COMP_INTEL_ANALYZE_RECENCY_DAYS", 180),
            documents_per_query=_env_int("COMP_INTEL_ANALYZE_DOCS_PER_QUERY", 6),
            max_documents_per_competitor=_env_int("COMP_INTEL_ANALYZE_MAX_DOCS_PER_COMPETITOR", 10),
            max_scraped_urls_per_competitor=_env_int("COMP_INTEL_ANALYZE_MAX_SCRAPED_URLS", 3),
            use_cache=True,
        )
        response = _run_async(service.generate(request))
        competitive_context = _context_from_external(
            state=state,
            existing=existing_context,
            response=response,
            cache_hit=False,
        )
        _run_async(_COMP_CONTEXT_CACHE.set(key, dict(competitive_context)))
        return {**state, "competitive_context": competitive_context}
    except Exception as exc:  # noqa: BLE001
        _logger.warning("competitor_node external fetch failed: %s", exc, exc_info=True)
        fallback = {
            **existing_context,
            "source": "deterministic_local" if existing_context.get("available") else "unavailable",
            "generated_at": _now_iso(),
            "warnings": [*existing_context.get("warnings", []), "external_fetch_failed"],
        }
        return {**state, "competitive_context": fallback}
