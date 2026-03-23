"""POST /api/business-intelligence — full intelligence pipeline endpoint.

Chains:
    1. ContextAnalyzer   → BusinessContext
    2. IntelligenceOrchestrator (search + news + trends) → IntelligenceBundle
    3. InsightSynthesizer → InsightBlock
    4. StrategyGenerator  → StrategyBlock

Returns a unified JSON response with context, insights, strategy, and
composite confidence.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, List, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.business_intelligence.context_analyzer import (
    BusinessContext,
    ContextAnalyzer,
    ContextAnalyzerError,
)
from app.business_intelligence.insight_synthesizer import (
    InsightBlock,
    InsightSynthesizer,
    InsightSynthesizerError,
)
from app.business_intelligence.intelligence_orchestrator import (
    IntelligenceBundle,
    IntelligenceOrchestrator,
)
from app.business_intelligence.strategy_generator import (
    StrategyBlock,
    StrategyGenerator,
    StrategyGeneratorError,
)
from app.security.dependencies import require_security_context

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["business-intelligence"],
    dependencies=[Depends(require_security_context)],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class BusinessIntelligenceRequest(BaseModel):
    """Input contract for the business intelligence endpoint."""

    model_config = ConfigDict(extra="forbid")

    business_prompt: str = Field(
        min_length=5,
        max_length=2000,
        description="Free-text business description (e.g. 'AI SaaS for healthcare clinics in India').",
    )


class PipelineStageResult(BaseModel):
    """Status of one pipeline stage in the response."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    status: Literal["success", "failed", "skipped"]
    duration_ms: float = Field(ge=0.0)
    error: str | None = None


class BusinessIntelligenceResponse(BaseModel):
    """Output contract matching the React frontend's expected shape."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "partial", "failed"]
    context: BusinessContext | None = None
    insights: InsightBlock | None = None
    strategy: StrategyBlock | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    pipeline: List[PipelineStageResult] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    generated_at: datetime


# ---------------------------------------------------------------------------
# Service construction (lazy, cached)
# ---------------------------------------------------------------------------


def _build_llm_adapter():
    """Build the appropriate LLM adapter based on environment."""
    from llm_synthesis.adapter import BaseLLMAdapter, MockLLMAdapter, OpenAILLMAdapter

    adapter_type = os.getenv("LLM_ADAPTER", "openai").strip().lower()
    if adapter_type == "mock":
        return MockLLMAdapter()

    model = (
        str(os.getenv("LLM_MODEL", "gpt-5.4") or "").strip()
        or "gpt-5.4"
    )
    api_key = os.getenv("LLM_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    return OpenAILLMAdapter(model=model, api_key=api_key or None)


def _build_orchestrator_deps():
    """Wire up the intelligence orchestrator dependencies from environment."""
    from app.competitor_intelligence.config import CompetitorIntelligenceConfig
    from app.competitor_intelligence.factory import (
        _build_extractor,
        _build_search_provider,
    )
    from app.competitor_intelligence.scraping import RequestsScraper

    cfg = CompetitorIntelligenceConfig.from_env()
    search_provider = _build_search_provider(cfg)
    scraper = RequestsScraper(config=cfg.scraper)
    extractor = _build_extractor(cfg)

    # Optional connectors — gracefully skip if not configured
    news_connector = _build_news_connector()
    trends_connector = _build_trends_connector()

    return cfg, search_provider, scraper, extractor, news_connector, trends_connector


def _build_news_connector():
    """Build news connector if configured, otherwise None."""
    try:
        from app.config import get_external_http_settings, get_news_api_settings
        from app.connectors.news_api_connector import NewsAPIConnector

        settings = get_news_api_settings()
        if not settings.enabled or not settings.api_key:
            return None
        return NewsAPIConnector(
            settings=settings,
            http_settings=get_external_http_settings(),
        )
    except Exception:
        logger.debug("News connector not available", exc_info=True)
        return None


def _build_trends_connector():
    """Build Google Trends connector if configured, otherwise None."""
    try:
        from app.config import get_external_http_settings, get_google_trends_settings
        from app.connectors.google_trends_connector import GoogleTrendsConnector

        settings = get_google_trends_settings()
        if not settings.enabled:
            return None
        return GoogleTrendsConnector(
            settings=settings,
            http_settings=get_external_http_settings(),
        )
    except Exception:
        logger.debug("Trends connector not available", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/business-intelligence",
    response_model=BusinessIntelligenceResponse,
    summary="Run full business intelligence pipeline",
    responses={
        422: {"description": "Validation error in request body"},
        500: {"description": "Internal pipeline failure"},
    },
)
async def run_business_intelligence(
    body: BusinessIntelligenceRequest,
) -> BusinessIntelligenceResponse:
    """Execute the full BI pipeline: context → search → synthesize → strategize.

    Every stage is fault-tolerant — partial results are returned with
    degraded confidence rather than a 500 error.
    """
    started_at = time.perf_counter()
    pipeline: list[PipelineStageResult] = []
    warnings: list[str] = []

    # -- Build services -----------------------------------------------------
    try:
        adapter = _build_llm_adapter()
        cfg, search_provider, scraper, extractor, news_conn, trends_conn = (
            _build_orchestrator_deps()
        )
    except Exception as exc:
        logger.exception("business_intelligence: service construction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "internal_failure",
                "message": f"Service initialization failed: {exc}",
            },
        ) from exc

    context_analyzer = ContextAnalyzer(adapter)
    orchestrator = IntelligenceOrchestrator(
        context_analyzer=context_analyzer,
        search_provider=search_provider,
        scraper=scraper,
        extractor=extractor,
        news_connector=news_conn,
        trends_connector=trends_conn,
        config=cfg,
    )
    synthesizer = InsightSynthesizer(adapter)
    strategy_gen = StrategyGenerator(adapter)

    # -- Stage 1+2: Orchestrator (context + search + news + trends) ---------
    bundle: IntelligenceBundle | None = None
    t0 = time.perf_counter()
    try:
        bundle = await orchestrator.run(body.business_prompt)
        pipeline.append(PipelineStageResult(
            stage="orchestration",
            status="success" if bundle.status == "success" else "failed" if bundle.status == "failed" else "success",
            duration_ms=_elapsed(t0),
        ))
        warnings.extend(bundle.warnings)
    except Exception as exc:
        logger.exception("business_intelligence: orchestration failed")
        pipeline.append(PipelineStageResult(
            stage="orchestration",
            status="failed",
            duration_ms=_elapsed(t0),
            error=str(exc),
        ))
        warnings.append(f"orchestration: {exc}")

    # -- Stage 3: Insight synthesis -----------------------------------------
    insight_block: InsightBlock | None = None
    t0 = time.perf_counter()
    if bundle and bundle.signals:
        try:
            insight_block = synthesizer.synthesize(bundle)
            pipeline.append(PipelineStageResult(
                stage="synthesis",
                status="success",
                duration_ms=_elapsed(t0),
            ))
        except (InsightSynthesizerError, ValueError) as exc:
            logger.warning("business_intelligence: synthesis failed — %s", exc)
            pipeline.append(PipelineStageResult(
                stage="synthesis",
                status="failed",
                duration_ms=_elapsed(t0),
                error=str(exc),
            ))
            warnings.append(f"synthesis: {exc}")
    else:
        pipeline.append(PipelineStageResult(
            stage="synthesis",
            status="skipped",
            duration_ms=_elapsed(t0),
            error="No signals available from orchestration.",
        ))

    # -- Stage 4: Strategy generation ---------------------------------------
    strategy_block: StrategyBlock | None = None
    t0 = time.perf_counter()
    if insight_block:
        try:
            strategy_block = strategy_gen.generate(insight_block)
            pipeline.append(PipelineStageResult(
                stage="strategy",
                status="success",
                duration_ms=_elapsed(t0),
            ))
        except (StrategyGeneratorError, ValueError) as exc:
            logger.warning("business_intelligence: strategy failed — %s", exc)
            pipeline.append(PipelineStageResult(
                stage="strategy",
                status="failed",
                duration_ms=_elapsed(t0),
                error=str(exc),
            ))
            warnings.append(f"strategy: {exc}")
    else:
        pipeline.append(PipelineStageResult(
            stage="strategy",
            status="skipped",
            duration_ms=_elapsed(t0),
            error="No insight block available from synthesis.",
        ))

    # -- Compose response ---------------------------------------------------
    confidence = _composite_confidence(bundle, insight_block, strategy_block)

    has_any_failure = any(s.status == "failed" for s in pipeline)
    has_any_success = any(s.status == "success" for s in pipeline)
    if not has_any_success:
        overall_status: Literal["success", "partial", "failed"] = "failed"
    elif has_any_failure:
        overall_status = "partial"
    else:
        overall_status = "success"

    elapsed_ms = _elapsed(started_at)
    logger.info(
        "business_intelligence_complete status=%s confidence=%.3f elapsed_ms=%s",
        overall_status, confidence, elapsed_ms,
    )

    return BusinessIntelligenceResponse(
        status=overall_status,
        context=bundle.business_context if bundle else None,
        insights=insight_block,
        strategy=strategy_block,
        confidence=round(confidence, 4),
        pipeline=pipeline,
        warnings=warnings,
        generated_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _elapsed(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 2)


def _composite_confidence(
    bundle: IntelligenceBundle | None,
    insight: InsightBlock | None,
    strategy: StrategyBlock | None,
) -> float:
    """Weighted composite: 40% bundle + 35% insight + 25% strategy."""
    scores: list[tuple[float, float]] = []
    if bundle is not None:
        scores.append((bundle.confidence, 0.40))
    if insight is not None:
        scores.append((insight.confidence, 0.35))
    if strategy is not None:
        scores.append((strategy.confidence, 0.25))

    if not scores:
        return 0.0

    total_weight = sum(w for _, w in scores)
    if total_weight == 0:
        return 0.0
    return sum(s * w for s, w in scores) / total_weight
