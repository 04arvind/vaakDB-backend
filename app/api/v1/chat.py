"""POST /chat: runs the chat graph.

Default response is Server-Sent Events:
    event: status   data: {"stage": "route_intent", "message": "Understanding your request"}
    ...
    event: result   data: {"type": "...", "data": {...}, "message": "...", "conversation_id": "..."}
 or event: error    data: {"type": "error", "data": {"code": ..., "message": ...}, ...}

With ?stream=false (or Accept: application/json) the final envelope is returned as JSON.
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from app.api.deps import CurrentCaller, DbSession
from app.core.exceptions import AppError, ErrorCode
from app.repositories import conversations
from app.schemas.chat import ChatRequest, ConversationOut
from app.services import chat_service
from app.services.chat_service import error_envelope

log = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


def get_chat_deps(request: Request) -> chat_service.ChatDeps | None:
    """Tests (or the app) can set app.state.chat_deps to inject a fake LLM."""
    return getattr(request.app.state, "chat_deps", None)


@router.post("/chat", response_model=None)
async def chat(
    body: ChatRequest,
    request: Request,
    db: DbSession,
    caller: CurrentCaller,
    stream: Annotated[bool | None, Query(description="false = plain JSON response")] = None,
) -> EventSourceResponse | JSONResponse:
    conv = conversations.get_or_create(db, caller.user_id, body.conversation_id,
                                       title=body.message[:80])
    conversation_id = str(conv.id)
    upload_id = str(body.upload_id) if body.upload_id else None
    deps = get_chat_deps(request)
    wants_json = stream is False or (
        stream is None and "text/event-stream" not in request.headers.get("accept", "")
        and "application/json" in request.headers.get("accept", "")
    )

    def finish(envelope: dict[str, Any]) -> dict[str, Any]:
        return {**envelope, "conversation_id": conversation_id}

    if wants_json:
        try:
            envelope = await chat_service.run_chat(caller, body.message, upload_id=upload_id,
                                                   conversation_id=conversation_id, deps=deps)
        except AppError as exc:
            envelope = error_envelope(exc)
        return JSONResponse(finish(envelope))

    async def events() -> AsyncIterator[dict[str, str]]:
        try:
            async for event, payload in chat_service.stream_chat(
                caller, body.message, upload_id=upload_id, conversation_id=conversation_id,
                deps=deps,
            ):
                if event != "status":
                    payload = finish(payload)
                yield {"event": event, "data": json.dumps(payload, default=str)}
        except Exception as exc:  # never leak internals into the stream
            if not isinstance(exc, AppError):
                log.exception("chat stream failed")
            err = exc if isinstance(exc, AppError) else AppError(ErrorCode.INTERNAL_ERROR)
            yield {"event": "error", "data": json.dumps(finish(error_envelope(err)))}

    return EventSourceResponse(events(), ping=15)


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(db: DbSession, caller: CurrentCaller,
                       limit: Annotated[int, Query(ge=1, le=200)] = 50) -> list[ConversationOut]:
    return [
        ConversationOut(conversation_id=str(c.id), title=c.title,
                        created_at=c.created_at.isoformat() if c.created_at else None)
        for c in conversations.list_for_user(db, caller.user_id, limit)
    ]
