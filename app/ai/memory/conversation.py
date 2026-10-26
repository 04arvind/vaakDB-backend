"""Chat history stored in PostgreSQL.

Every function takes user_id and enforces ownership in SQL, so a user who
guesses or steals another user's session_id still cannot read or write it. A
session that belongs to someone else raises SessionNotFoundError, exactly like
a missing one (no existence leak).
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

from sqlalchemy import text

from app.ai.db.engine import get_write_engine
from app.ai.exceptions import SessionNotFoundError
from app.ai.schemas.llm_io import Message

_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_CONTENT_CHARS = 20_000


def new_session_id() -> str:
    return uuid.uuid4().hex


async def ensure_session(user_id: str, session_id: str | None = None) -> str:
    """Return a usable session id: creates it if new, verifies the owner if it exists."""
    sid = session_id or new_session_id()
    if not _SESSION_RE.match(sid):
        raise SessionNotFoundError("Invalid session id.")
    if not user_id:
        raise ValueError("A user is required.")

    async with get_write_engine().begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO ai_conversations (id, user_id) VALUES (:sid, :uid) "
                "ON CONFLICT (id) DO NOTHING"
            ),
            {"sid": sid, "uid": user_id},
        )
        owner = (
            await conn.execute(
                text("SELECT user_id FROM ai_conversations WHERE id = :sid"), {"sid": sid}
            )
        ).scalar_one()

    if owner != user_id:
        raise SessionNotFoundError()
    return sid


async def add_exchange(
    session_id: str,
    user_id: str,
    user_text: str,
    assistant_text: str,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    """Save a user message and the assistant reply together (one transaction).

    meta is stored on the assistant message (e.g. provider, model, tokens, sql).
    """
    title = " ".join(user_text.split())[:80] or None
    async with get_write_engine().begin() as conn:
        await _append(conn, session_id, user_id, "user", user_text, None)
        await _append(conn, session_id, user_id, "assistant", assistant_text, meta)
        await conn.execute(
            text(
                "UPDATE ai_conversations SET updated_at = now(), "
                "title = COALESCE(title, CAST(:title AS TEXT)) WHERE id = :sid"
            ),
            {"sid": session_id, "title": title},
        )


async def add_message(
    session_id: str, user_id: str, role: str, content: str,
    *, meta: dict[str, Any] | None = None,
) -> None:
    """Save a single message (use add_exchange for the normal request/response flow)."""
    async with get_write_engine().begin() as conn:
        await _append(conn, session_id, user_id, role, content, meta)
        await conn.execute(
            text("UPDATE ai_conversations SET updated_at = now() WHERE id = :sid"),
            {"sid": session_id},
        )


async def get_history(session_id: str, user_id: str, *, limit: int = 20) -> list[Message]:
    """Most recent `limit` messages in chronological order, ready for the LLM."""
    limit = max(1, min(limit, 200))
    async with get_write_engine().connect() as conn:
        rows = (
            await conn.execute(
                text(
                    """SELECT m.role, m.content
                       FROM ai_messages m
                       JOIN ai_conversations c ON c.id = m.conversation_id
                       WHERE m.conversation_id = :sid AND c.user_id = :uid
                       ORDER BY m.id DESC LIMIT :n"""
                ),
                {"sid": session_id, "uid": user_id, "n": limit},
            )
        ).all()
    return [Message(role=r.role, content=r.content) for r in reversed(rows)]


async def get_history_records(
    session_id: str, user_id: str, *, limit: int = 100
) -> list[dict[str, Any]]:
    """Same as get_history but with timestamps, for a GET /ai/history endpoint."""
    limit = max(1, min(limit, 500))
    async with get_write_engine().connect() as conn:
        rows = (
            await conn.execute(
                text(
                    """SELECT m.role, m.content, m.created_at
                       FROM ai_messages m
                       JOIN ai_conversations c ON c.id = m.conversation_id
                       WHERE m.conversation_id = :sid AND c.user_id = :uid
                       ORDER BY m.id DESC LIMIT :n"""
                ),
                {"sid": session_id, "uid": user_id, "n": limit},
            )
        ).all()
    return [
        {"role": r.role, "content": r.content, "created_at": r.created_at.isoformat()}
        for r in reversed(rows)
    ]


async def list_sessions(user_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 200))
    async with get_write_engine().connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT id, title, updated_at FROM ai_conversations "
                    "WHERE user_id = :uid ORDER BY updated_at DESC LIMIT :n"
                ),
                {"uid": user_id, "n": limit},
            )
        ).all()
    return [
        {"session_id": r.id, "title": r.title, "updated_at": r.updated_at.isoformat()}
        for r in rows
    ]


async def delete_session(session_id: str, user_id: str) -> bool:
    """Delete a conversation and all its messages. True if something was deleted."""
    async with get_write_engine().begin() as conn:
        result = await conn.execute(
            text("DELETE FROM ai_conversations WHERE id = :sid AND user_id = :uid"),
            {"sid": session_id, "uid": user_id},
        )
    return result.rowcount > 0


# --------------------------------------------------------------------------- #
async def _append(conn, session_id: str, user_id: str, role: str, content: str,
                  meta: dict[str, Any] | None) -> None:
    if role not in ("user", "assistant"):
        raise ValueError("Role must be 'user' or 'assistant'.")

    # INSERT ... WHERE EXISTS: the row is only written if the session belongs to user_id.
    result = await conn.execute(
        text(
            """INSERT INTO ai_messages (conversation_id, role, content, meta)
               SELECT CAST(:sid AS TEXT), CAST(:role AS TEXT), CAST(:content AS TEXT),
                      CAST(CAST(:meta AS TEXT) AS jsonb)
               WHERE EXISTS (
                 SELECT 1 FROM ai_conversations WHERE id = :sid AND user_id = :uid
               )"""
        ),
        {
            "sid": session_id,
            "uid": user_id,
            "role": role,
            "content": content[:MAX_CONTENT_CHARS],
            "meta": json.dumps(meta or {}, default=str),
        },
    )
    if result.rowcount != 1:
        raise SessionNotFoundError()