from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class QueryHistoryOut(BaseModel):
    id: int
    conversation_id: str | None
    message: str
    intent: str | None
    response_type: str | None
    status: str
    error_code: str | None
    sql: str | None
    operation_id: str | None
    created_at: str
    response: dict[str, Any] | None = None
