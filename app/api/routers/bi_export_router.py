"""
app/api/routers/bi_export_router.py

PowerBI-compatible export endpoint.

GET /export/bi

Query parameters
----------------
dataset       : "records" | "kpis" | "forecasts" | "risk"  (default: "records")
output_format : "csv" | "json"                              (default: "csv")
entity_name   : optional exact-match entity filter
date_from     : optional ISO date lower bound  (YYYY-MM-DD, inclusive)
date_to       : optional ISO date upper bound  (YYYY-MM-DD, inclusive)
limit         : max rows returned, 1–100 000   (default: 10 000)

Responses
---------
CSV  → StreamingResponse, Content-Type: text/csv
       Content-Disposition: attachment; filename=<dataset>_export.csv
JSON → JSONResponse, Content-Type: application/json
       Body: {"dataset": str, "rows": int, "data": list[dict]}

All transformation logic lives in BIExportService; the router only handles
HTTP plumbing (serialisation, content-type, error mapping).
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from sqlalchemy.orm import Session

from agent.graph import insight_graph
from agent.nodes.node_result import payload_of
from app.failure_codes import INTERNAL_FAILURE, SCHEMA_CONFLICT, build_error_detail
from app.security.dependencies import (
    assert_entity_allowed_for_tenant,
    require_security_context,
)
from app.security.models import SecurityContext
from app.services.competitive_benchmark_service import build_competitive_benchmark_snapshot
from app.services.bi_export_service import (
    BIExportService,
    ExportResult,
    get_bi_export_service,
)
from app.services.bi_workbook_export import generate_bi_workbook
from app.services.category_registry import get_processing_strategy
from llm_synthesis.schema import InsightOutput
from db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["export"])

_VALID_DATASETS = frozenset({"records", "kpis", "forecasts", "risk"})
_VALID_FORMATS = frozenset({"csv", "json"})


# ---------------------------------------------------------------------------
# Serialisation helpers (no business logic)
# ---------------------------------------------------------------------------


def _to_csv_streaming(result: ExportResult, filename: str) -> StreamingResponse:
    """Stream *result* as a UTF-8 CSV file download."""

    def _generate() -> io.Iterator[str]:
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=result.fields,
            extrasaction="ignore",
            restval="",
            lineterminator="\r\n",
        )
        writer.writeheader()
        yield buf.getvalue()

        for row in result.rows:
            buf.seek(0)
            buf.truncate(0)
            # Normalise None → "" for PowerBI compatibility
            clean = {k: ("" if v is None else v) for k, v in row.items()}
            writer.writerow(clean)
            yield buf.getvalue()

    return StreamingResponse(
        content=_generate(),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Row-Count": str(len(result.rows)),
        },
    )


def _to_json_response(result: ExportResult, dataset: str) -> JSONResponse:
    """Return *result* as a structured JSON response."""
    return JSONResponse(
        content={
            "dataset": dataset,
            "rows": len(result.rows),
            "fields": result.fields,
            "data": result.rows,
        }
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get("/export/bi", summary="Export analytics data for BI tools", response_model=None)
@router.get("/export/powerbi", summary="Export analytics data for BI tools", response_model=None)
def export_bi(
    dataset: str = Query(
        default="records",
        description='Dataset to export: "records", "kpis", "forecasts", or "risk".',
    ),
    output_format: str = Query(
        default="csv",
        alias="format",
        description='Output format: "csv" (file download) or "json".',
    ),
    entity_name: str | None = Query(
        default=None,
        description="Filter by exact entity name.",
    ),
    date_from: date | None = Query(
        default=None,
        description="Inclusive start date filter (YYYY-MM-DD).",
    ),
    date_to: date | None = Query(
        default=None,
        description="Inclusive end date filter (YYYY-MM-DD).",
    ),
    limit: int = Query(
        default=10_000,
        ge=1,
        le=100_000,
        description="Maximum number of rows to return.",
    ),
    db: Session = Depends(get_db),
    service: BIExportService = Depends(get_bi_export_service),
    security: SecurityContext = Depends(require_security_context),
) -> StreamingResponse | JSONResponse:
    """
    Export analytics data in a flat, tabular format consumable by PowerBI
    (or any other BI tool) via CSV file download or JSON response.

    Use the ``dataset`` parameter to choose the data source and ``format``
    to choose the output encoding.  Apply ``entity_name``, ``date_from``,
    and ``date_to`` to narrow the result set before the row ``limit`` is
    applied.
    """
    # --- Validate query params ---
    if entity_name:
        assert_entity_allowed_for_tenant(entity_name=entity_name, security=security)
    if dataset not in _VALID_DATASETS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=build_error_detail(
                code=SCHEMA_CONFLICT,
                message=f"Invalid dataset {dataset!r}. Must be one of: {sorted(_VALID_DATASETS)}.",
            ),
        )
    if output_format not in _VALID_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=build_error_detail(
                code=SCHEMA_CONFLICT,
                message=f"Invalid format {output_format!r}. Must be one of: {sorted(_VALID_FORMATS)}.",
            ),
        )
    if date_from and date_to and date_from > date_to:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=build_error_detail(
                code=SCHEMA_CONFLICT,
                message="date_from must not be later than date_to.",
            ),
        )

    # --- Delegate all data work to the service ---
    try:
        result: ExportResult = service.export(
            db,
            dataset=dataset,
            entity_name=entity_name,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=build_error_detail(
                code=SCHEMA_CONFLICT,
                message=str(exc),
            ),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("BI export failed dataset=%r entity=%r", dataset, entity_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(
                code=INTERNAL_FAILURE,
                message="Export failed; see server logs for details.",
            ),
        ) from exc

    logger.info(
        "BI export dataset=%r format=%r entity=%r rows=%d",
        dataset,
        output_format,
        entity_name,
        len(result.rows),
    )

    # --- Serialise ---
    filename = f"{dataset}_export.csv"
    if output_format == "csv":
        return _to_csv_streaming(result, filename)
    return _to_json_response(result, dataset)


@router.get("/export/report", summary="Export formatted insight report", response_model=None)
def export_report(
    entity_name: str = Query(..., description="Entity/client identifier."),
    business_type: str | None = Query(
        default=None,
        description="Optional business type/category override.",
    ),
    prompt: str = Query(
        default="Generate deterministic insight report.",
        description="Prompt used for synthesis.",
    ),
    output_format: str = Query(
        default="json",
        alias="format",
        description='Output format: "json" or "md".',
    ),
    competitors: str | None = Query(
        default=None,
        description="Comma-separated competitor names for benchmarking.",
    ),
    self_analysis_only: bool = Query(
        default=False,
        description="When true, force data-only self analysis and disable competitor benchmarking in synthesis.",
    ),
    db: Session = Depends(get_db),
    security: SecurityContext = Depends(require_security_context),
) -> JSONResponse | PlainTextResponse:
    assert_entity_allowed_for_tenant(entity_name=entity_name, security=security)
    if output_format not in {"json", "md"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=build_error_detail(
                code=SCHEMA_CONFLICT,
                message="Invalid format for report export. Use 'json' or 'md'.",
            ),
        )

    try:
        resolved = get_processing_strategy(business_type) or "general_timeseries"
        invoke_state: dict[str, Any] = {
            "request_id": security.request_id,
            "user_query": prompt,
            "business_type": resolved,
            "entity_name": entity_name,
        }
        if competitors:
            invoke_state["competitors"] = [
                name.strip() for name in competitors.split(",") if name.strip()
            ]
        if self_analysis_only:
            invoke_state["self_analysis_only"] = True
        state = insight_graph.invoke(invoke_state)
        response_raw = state.get("final_response")
        if not isinstance(response_raw, str):
            raise ValueError("Graph did not return final_response.")

        insight_payload = InsightOutput.model_validate(json.loads(response_raw))
        competitive = insight_payload.competitive_analysis
        strategic = insight_payload.strategic_recommendations
        immediate = strategic.immediate_actions[0] if strategic.immediate_actions else ""
        pipeline_status = str(state.get("pipeline_status") or insight_payload.pipeline_status)
        try:
            benchmark_snapshot = build_competitive_benchmark_snapshot(
                db=db,
                entity_name=entity_name,
                business_type=resolved,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Report benchmark snapshot failed: %s", exc)
            benchmark_snapshot = {
                "status": "partial",
                "reason": "benchmark_snapshot_unavailable",
            }
        report_payload: dict[str, Any] = {
            "entity_name": entity_name,
            "business_type": resolved,
            "pipeline_status": pipeline_status,
            "confidence_score": competitive.confidence,
            "insight_payload": insight_payload.model_dump(),
            "summary_snapshot": {
                "summary": competitive.summary,
                "market_position": competitive.market_position,
                "relative_performance": competitive.relative_performance,
                "recommended_focus": immediate,
            },
            "derived_signals": {
                "growth": payload_of(state.get("growth_data")) or {},
                "timeseries_factors": payload_of(state.get("timeseries_factors_data")) or {},
                "cohort": payload_of(state.get("cohort_data")) or {},
                "category_formula": payload_of(state.get("category_formula_data")) or {},
                "unit_economics": payload_of(state.get("unit_economics_data")) or {},
                "multivariate_scenario": payload_of(state.get("multivariate_scenario_data")) or {},
                "role_contribution": payload_of(state.get("segmentation")) or {},
                "risk": payload_of(state.get("risk_data")) or {},
                "prioritization": state.get("prioritization") or {},
                "competitive_benchmark": benchmark_snapshot,
            },
        }

        if output_format == "json":
            return JSONResponse(content=report_payload)

        md = _report_markdown(report_payload)
        return PlainTextResponse(content=md, media_type="text/markdown; charset=utf-8")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Report export failed entity=%r", entity_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(
                code=INTERNAL_FAILURE,
                message=f"Report export failed: {exc}",
            ),
        ) from exc


@router.get("/export/bi-workbook", summary="Export interactive BI workbook (.xlsx)", response_model=None)
def export_bi_workbook(
    entity_name: str = Query(..., description="Entity/client identifier."),
    business_type: str | None = Query(
        default=None,
        description="Optional business type/category override.",
    ),
    prompt: str = Query(
        default="Generate deterministic insight report.",
        description="Prompt used for synthesis.",
    ),
    db: Session = Depends(get_db),
    security: SecurityContext = Depends(require_security_context),
) -> StreamingResponse:
    """Export a multi-sheet Excel workbook with charts, KPI trends, risk
    scores, forecasts, and strategic insights.  Openable in Power BI,
    Excel, or any tool that supports .xlsx."""
    try:
        assert_entity_allowed_for_tenant(entity_name=entity_name, security=security)
        resolved = get_processing_strategy(business_type) or "general_timeseries"
        state = insight_graph.invoke(
            {
                "request_id": security.request_id,
                "user_query": prompt,
                "business_type": resolved,
                "entity_name": entity_name,
            }
        )

        workbook_bytes = generate_bi_workbook(
            db,
            entity_name=entity_name,
            business_type=resolved,
            graph_state=state,
        )

        filename = f"{entity_name}_insight_workbook.xlsx"
        return StreamingResponse(
            content=io.BytesIO(workbook_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("BI workbook export failed entity=%r", entity_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(
                code=INTERNAL_FAILURE,
                message=f"Workbook export failed: {exc}",
            ),
        ) from exc


def _report_markdown(payload: dict[str, Any]) -> str:
    insight = payload.get("insight_payload") or {}
    competitive = insight.get("competitive_analysis") if isinstance(insight, dict) else {}
    strategic = insight.get("strategic_recommendations") if isinstance(insight, dict) else {}
    derived = payload.get("derived_signals") or {}
    immediate = strategic.get("immediate_actions") if isinstance(strategic, dict) else []
    mid_term = strategic.get("mid_term_moves") if isinstance(strategic, dict) else []
    defensive = strategic.get("defensive_strategies") if isinstance(strategic, dict) else []
    offensive = strategic.get("offensive_strategies") if isinstance(strategic, dict) else []
    lines = [
        f"# Insight Report: {payload.get('entity_name')}",
        "",
        f"- Business Type: {payload.get('business_type')}",
        f"- Pipeline Status: {payload.get('pipeline_status')}",
        f"- Confidence Score: {payload.get('confidence_score')}",
        "",
        "## Competitive Analysis",
        f"- Summary: {competitive.get('summary') if isinstance(competitive, dict) else ''}",
        f"- Market Position: {competitive.get('market_position') if isinstance(competitive, dict) else ''}",
        f"- Relative Performance: {competitive.get('relative_performance') if isinstance(competitive, dict) else ''}",
        f"- Key Advantages: {competitive.get('key_advantages') if isinstance(competitive, dict) else []}",
        f"- Key Vulnerabilities: {competitive.get('key_vulnerabilities') if isinstance(competitive, dict) else []}",
        "",
        "## Strategic Recommendations",
        f"- Immediate Actions: {immediate}",
        f"- Mid-Term Moves: {mid_term}",
        f"- Defensive Strategies: {defensive}",
        f"- Offensive Strategies: {offensive}",
        "",
        "## Derived Signals",
        "```json",
        json.dumps(derived, indent=2, default=str),
        "```",
    ]
    return "\n".join(lines)
