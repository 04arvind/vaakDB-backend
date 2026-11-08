"""Pending operations of the caller. Every endpoint only sees the caller's own
operations (another user's operation id gives 404)."""
from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Path, status
from pydantic import BaseModel

from app.api.deps import CurrentCaller, DbSession
from app.db.target_db import get_read_engine
from app.schemas.plans import Plan
from app.services import operations_service
from app.services.chat_service import preview_message
from app.services.plan_validator import validate_plan

router = APIRouter(prefix="/operations", tags=["Operations"])

OperationId = Annotated[uuid.UUID, Path(description="operation_id from an operation_preview")]


class PlanRequest(BaseModel):
    plan: Plan


@router.post("", status_code=status.HTTP_201_CREATED)
def create_operation(body: PlanRequest, db: DbSession, caller: CurrentCaller) -> dict[str, Any]:
    """Create a pending operation from a structured plan (no LLM), e.g. the
    "Create table" form. Same validation and preview -> confirm flow as chat."""
    with get_read_engine().connect() as conn:
        validated = validate_plan(body.plan, caller=caller, conn=conn)
    op = operations_service.create_pending(db, caller=caller, validated=validated)
    preview = dict(op.preview_json)
    return {"type": "operation_preview", "data": preview, "message": preview_message(preview)}


@router.get("/{operation_id}")
def get_operation(operation_id: OperationId, db: DbSession, caller: CurrentCaller) -> dict[str, Any]:
    """Status, preview and (after confirmation) the result."""
    return operations_service.get_operation(db, operation_id, caller)


@router.post("/{operation_id}/confirm")
def confirm_operation(operation_id: OperationId, db: DbSession,
                      caller: CurrentCaller) -> dict[str, Any]:
    """Execute a pending operation (same user only). Idempotent: confirming an
    executed operation returns its stored result without running anything."""
    return operations_service.confirm(db, operation_id, caller)


@router.post("/{operation_id}/cancel")
def cancel_operation(operation_id: OperationId, db: DbSession,
                     caller: CurrentCaller) -> dict[str, Any]:
    view = operations_service.cancel(db, operation_id, caller)
    return {"type": "message", "data": view, "message": "Operation cancelled. Nothing was changed."}
