"""
app/api/routers/client_router.py

Client management endpoints.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from db.models.client import Client
from db.session import get_db

router = APIRouter(prefix="/clients", tags=["clients"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ClientCreateRequest(BaseModel):
    name: str
    domain: str | None = None


class ClientResponse(BaseModel):
    id: uuid.UUID
    name: str
    domain: str | None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=ClientResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_client(
    body: ClientCreateRequest,
    db: Session = Depends(get_db),
) -> ClientResponse:
    """
    Create a new client record.

    Raises HTTP 409 if a client with the same name already exists.
    """
    client = Client(name=body.name, domain=body.domain)
    db.add(client)
    try:
        db.commit()
        db.refresh(client)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A client with name {body.name!r} already exists.",
        )
    return ClientResponse.model_validate(client)
