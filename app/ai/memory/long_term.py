"""Long-term memories: durable facts and preferences per user.

Flow (run it after the reply is sent, e.g. as a FastAPI BackgroundTask):
    memory_agent.extract_memories(history)  ->  save_memories(user_id, items)
or simply:
    await update_long_term_memory(user_id, session_id)

At answer time, call recall(user_id, question) and pass the result to
context_builder as a "What you know about this user" section.

Recall is by importance, recency and keyword overlap. When you have many memories per
user, upgrade it to semantic search with db/vector_store.py.
"""
from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any, Sequence

from sqlalchemy import text

from app.ai.db.engine import cfg, get_write_engine

if TYPE_CHECKING:  # avoid importing the agents (and providers) at module load
    from app.ai.agents.memory_agent import MemoryItem

log = logging.getLogger(__name__)

# Defence in depth: the memory agent is told not to save secrets, but never trust that alone.
_SECRET_PATTERNS = [
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),  # card-like numbers
    re.compile(r"\b(?:api[_ -]?key|password|passwd|secret|token|cvv|pin|otp)\b\s*(?:is|=|:)", re.I),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bgsk_[A-Za-z0-9]{16,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),  # JWT
]


def looks_sensitive(content: str) -> bool:
    return any(p.search(content) for p in _SECRET_PATTERNS)


async def save_memories(
    user_id: str,
    items: Sequence["MemoryItem"],
    *,
    source_session: str | None = None,
) -> int:
    """Store memories (duplicates just raise importance). Returns how many were accepted."""
    clean = [i for i in items if not looks_sensitive(i.content)]
    if len(clean) < len(items):
        log.info("Skipped %d memories that looked sensitive", len(items) - len(clean))
    if not clean:
        return 0

    cap = int(cfg("MAX_MEMORIES_PER_USER", 200))
    upsert = text(
        """INSERT INTO ai_memories (user_id, content, category, importance, source_session)
           VALUES (:uid, :content, :category, :importance, :src)
           ON CONFLICT (user_id, content) DO UPDATE
             SET importance = GREATEST(ai_memories.importance, EXCLUDED.importance),
                 updated_at = now()"""
    )
    trim = text(
        """DELETE FROM ai_memories
           WHERE user_id = :uid AND id IN (
             SELECT id FROM ai_memories WHERE user_id = :uid
             ORDER BY importance DESC, updated_at DESC OFFSET :cap
           )"""
    )
    params = [
        {
            "uid": user_id,
            "content": i.content.strip(),
            "category": i.category,
            "importance": i.importance,
            "src": source_session,
        }
        for i in clean
    ]
    async with get_write_engine().begin() as conn:
        await conn.execute(upsert, params)
        await conn.execute(trim, {"uid": user_id, "cap": cap})
    return len(clean)


async def recall(
    user_id: str, query: str | None = None, *, limit: int = 8, pool: int = 40
) -> list[str]:
    """Most relevant memories for this user, as plain sentences."""
    async with get_write_engine().connect() as conn:
        rows = (
            await conn.execute(
                text(
                    """SELECT content, importance FROM ai_memories WHERE user_id = :uid
                       ORDER BY importance DESC, updated_at DESC LIMIT :pool"""
                ),
                {"uid": user_id, "pool": pool},
            )
        ).all()

    if not query:
        return [r.content for r in rows[:limit]]

    words = set(re.findall(r"\w{3,}", query.lower()))

    def score(row) -> int:
        overlap = len(words & set(re.findall(r"\w{3,}", row.content.lower())))
        return row.importance + 2 * min(overlap, 3)

    ranked = sorted(rows, key=score, reverse=True)  # stable: ties keep recency order
    return [r.content for r in ranked[:limit]]


async def list_memories(user_id: str) -> list[dict[str, Any]]:
    async with get_write_engine().connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT id, content, category, importance FROM ai_memories "
                    "WHERE user_id = :uid ORDER BY importance DESC, updated_at DESC"
                ),
                {"uid": user_id},
            )
        ).all()
    return [
        {"id": r.id, "content": r.content, "category": r.category, "importance": r.importance}
        for r in rows
    ]


async def delete_memory(user_id: str, memory_id: int) -> bool:
    async with get_write_engine().begin() as conn:
        result = await conn.execute(
            text("DELETE FROM ai_memories WHERE id = :id AND user_id = :uid"),
            {"id": memory_id, "uid": user_id},
        )
    return result.rowcount > 0


async def delete_all_memories(user_id: str) -> int:
    """'Forget everything about me'."""
    async with get_write_engine().begin() as conn:
        result = await conn.execute(
            text("DELETE FROM ai_memories WHERE user_id = :uid"), {"uid": user_id}
        )
    return result.rowcount


async def update_long_term_memory(user_id: str, session_id: str, *, window: int = 10) -> int:
    """Look at the latest messages, let the memory agent pick what to keep, and save it.
    Best effort: failures are logged and never raised (this runs in the background)."""
    try:
        from app.ai.agents.memory_agent import MemoryAgent
        from app.ai.memory.conversation import get_history

        history = await get_history(session_id, user_id, limit=window)
        items = await MemoryAgent().extract_memories(history)
        return await save_memories(user_id, items, source_session=session_id)
    except Exception:
        log.exception("Long-term memory update failed")
        return 0