"""query_history (meta_db): one row per chat message and what it produced."""
from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.app_db import independent_session
from app.db.models import QueryHistory

log = logging.getLogger(__name__)

MAX_STORED_ROWS = 50


def _compact(envelope: dict[str, Any]) -> dict[str, Any]:
    """Keep history small: cap row lists stored with the response."""
    data = envelope.get("data")
    if isinstance(data, dict):
        data = dict(data)
        for key in ("rows", "sample_rows"):
            if isinstance(data.get(key), list) and len(data[key]) > MAX_STORED_ROWS:
                data[key] = data[key][:MAX_STORED_ROWS]
                data[f"{key}_truncated"] = True
        envelope = {**envelope, "data": data}
    return envelope


def record(
    *,
    user_id: int,
    conversation_id: uuid.UUID | None,
    message: str,
    intent: str | None,
    envelope: dict[str, Any],
    sql: str | None = None,
    operation_id: uuid.UUID | None = None,
) -> None:
    is_error = envelope.get("type") == "error"
    code = (envelope.get("data") or {}).get("code") if is_error else None
    try:
        with independent_session() as db:
            db.add(QueryHistory(
                user_id=user_id,
                conversation_id=conversation_id,
                message=message[:10_000],
                intent=intent,
                response_type=envelope.get("type"),
                sql=sql,
                operation_id=operation_id,
                status="error" if is_error else "ok",
                error_code=code,
                response_json=_compact(envelope),
            ))
    except Exception:
        log.exception("Could not record query history")


def list_for_user(db: Session, user_id: int, *, conversation_id: uuid.UUID | None = None,
                  limit: int = 50, offset: int = 0) -> list[QueryHistory]:
    stmt = select(QueryHistory).where(QueryHistory.user_id == user_id)
    if conversation_id is not None:
        stmt = stmt.where(QueryHistory.conversation_id == conversation_id)
    stmt = stmt.order_by(QueryHistory.created_at.desc(), QueryHistory.id.desc())
    return list(db.execute(stmt.limit(limit).offset(offset)).scalars())
