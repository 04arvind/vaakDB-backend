"""Query history of the current user (what was asked and what came back)."""
from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import CurrentCaller, DbSession
from app.repositories import query_runs
from app.schemas.query import QueryHistoryOut

router = APIRouter(prefix="/queries", tags=["Query history"])


@router.get("", response_model=list[QueryHistoryOut])
def history(
    db: DbSession,
    caller: CurrentCaller,
    conversation_id: uuid.UUID | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    include_response: bool = False,
) -> list[QueryHistoryOut]:
    rows = query_runs.list_for_user(db, caller.user_id, conversation_id=conversation_id,
                                    limit=limit, offset=offset)
    return [
        QueryHistoryOut(
            id=r.id,
            conversation_id=str(r.conversation_id) if r.conversation_id else None,
            message=r.message,
            intent=r.intent,
            response_type=r.response_type,
            status=r.status,
            error_code=r.error_code,
            sql=r.sql,
            operation_id=str(r.operation_id) if r.operation_id else None,
            created_at=r.created_at.isoformat(),
            response=r.response_json if include_response else None,
        )
        for r in rows
    ]
