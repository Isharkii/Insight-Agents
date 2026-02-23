"""
app/api/routers/analyze_router.py

Insight generation endpoint — exposes the full LangGraph pipeline as an API.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError
from sqlalchemy.orm import Session

from agent.graph import insight_graph
from agent.nodes.intent import intent_node
from app.services.csv_ingestion_service import (
    CSVIngestionService,
    get_csv_ingestion_service,
)
from db.session import get_db
from llm_synthesis.schema import InsightOutput

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analysis"])

_ALLOWED_BUSINESS_TYPES = {"saas", "ecommerce", "agency"}


@router.post("/analyze", response_model=InsightOutput)
def analyze(
    prompt: str = Form(..., description="Business prompt / user query"),
    file: Optional[UploadFile] = File(default=None, description="Optional CSV file"),
    client_id: Optional[str] = Form(default=None),
    business_type: Optional[str] = Form(default=None),
    model: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
    csv_service: CSVIngestionService = Depends(get_csv_ingestion_service),
) -> InsightOutput:
    """Run the full insight generation pipeline.

    Accepts an optional CSV file and query parameters. Internally:
    1. Extract intent from prompt
    2. Ingest CSV if present
    3. Invoke LangGraph pipeline
    4. Return structured InsightOutput
    """
    if model and model != "default":
        os.environ["LLM_MODEL"] = model

    try:
        # Step 1: Intent extraction
        seed_state = intent_node({"user_query": prompt})
        resolved_business_type = seed_state.get("business_type")
        resolved_entity_name = seed_state.get("entity_name")

        # Override with explicit params if provided
        if business_type and business_type in _ALLOWED_BUSINESS_TYPES:
            resolved_business_type = business_type

        ingest_business_type = (
            resolved_business_type
            if resolved_business_type in _ALLOWED_BUSINESS_TYPES
            else None
        )
        ingest_entity_name = resolved_entity_name or client_id or None

        # Step 2: CSV ingestion (if file provided)
        if file is not None:
            try:
                csv_service.ingest_csv(
                    upload_file=file,
                    db=db,
                    client_name=ingest_entity_name,
                    business_type=ingest_business_type,
                )
            finally:
                file.file.close()

        # Step 3: Invoke graph
        invoke_state: dict[str, Any] = {"user_query": prompt}
        if ingest_business_type:
            invoke_state["business_type"] = ingest_business_type
        if ingest_entity_name:
            invoke_state["entity_name"] = ingest_entity_name

        state = insight_graph.invoke(invoke_state)

        # Step 4: Extract and validate output
        response = state.get("final_response")
        if not isinstance(response, str):
            raise ValueError("Pipeline did not produce final_response.")

        payload = json.loads(response)
        return InsightOutput.model_validate(payload)

    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        logger.exception("Analysis pipeline failed")
        return InsightOutput.failure(str(exc))
    except Exception as exc:
        logger.exception("Analysis pipeline unexpected error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        ) from exc
