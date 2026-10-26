"""Long-term memory: durable facts about a user, stored in ai_memories.

    items = await MemoryAgent().extract_memories(messages)
    await save_memories(user_id, items, source_session=session_id)
    facts = await recall(user_id, query="monthly revenue", limit=5)

or, in one best-effort call after each exchange:
    await update_long_term_memory(user_id, session_id, messages)

Every query is scoped by user_id. Each user keeps at most MEMORY_MAX_PER_USER
memories; the least important / oldest are pruned first.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Sequence

from sqlalchemy import text

from app.ai.db.engine import cfg, get_write_engine
from app.ai.schemas.llm_io import Message

log = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]{3,}")
_RECALL_POOL = 100  # candidates scanned for keyword relevance

_UPSERT = text(
    """INSERT INTO ai_memories (user_id, content, category, importance, source_session)
       VALUES (:uid, :content, :category, :importance, :sid)
       ON CONFLICT (user_id, content) DO UPDATE
         SET importance = GREATEST(ai_memories.importance, EXCLUDED.importance),
             category   = EXCLUDED.category,
             updated_at = now()"""
)

_PRUNE = text(
    """DELETE FROM ai_memories
       WHERE user_id = :uid AND id NOT IN (
         SELECT id FROM ai_memories WHERE user_id = :uid
         ORDER BY importance DESC, updated_at DESC
         LIMIT :keep
       )"""
)


def _normalize(content: str) -> str:
    return " ".join(content.split()).strip()


async def save_memories(
    user_id: str,
    items: Sequence[Any],
    *,
    source_session: str | None = None,
) -> int:
    """Upsert MemoryItem-like objects (content, category, importance). Returns count saved."""
    if not user_id:
        raise ValueError("A user is required.")
    params = []
    seen: set[str] = set()
    for item in items:
        content = _normalize(getattr(item, "content", "") or "")[:500]
        if len(content) < 3 or content.lower() in seen:
            continue
        seen.add(content.lower())
        params.append({
            "uid": user_id,
            "content": content,
            "category": getattr(item, "category", "other") or "other",
            "importance": max(1, min(int(getattr(item, "importance", 3) or 3), 5)),
            "sid": source_session,
        })
    if not params:
        return 0

    keep = int(cfg("MEMORY_MAX_PER_USER", 200))
    async with get_write_engine().begin() as conn:
        await conn.execute(_UPSERT, params)
        await conn.execute(_PRUNE, {"uid": user_id, "keep": keep})
    log.info("Saved %d memories for user", len(params))
    return len(params)


async def recall(user_id: str, query: str | None = None, *, limit: int = 5) -> list[str]:
    """Most useful memories for this user, ranked by keyword overlap with `query`,
    then importance and recency. Returns plain sentences for the prompt."""
    limit = max(1, min(limit, 50))
    async with get_write_engine().connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT content, importance FROM ai_memories WHERE user_id = :uid "
                    "ORDER BY importance DESC, updated_at DESC LIMIT :n"
                ),
                {"uid": user_id, "n": _RECALL_POOL},
            )
        ).all()
    if not rows:
        return []

    words = set(_WORD.findall((query or "").lower()))
    ranked = sorted(
        enumerate(rows),
        key=lambda pair: (
            -len(words & set(_WORD.findall(pair[1].content.lower()))),
            -pair[1].importance,
            pair[0],  # recency order from SQL
        ),
    )
    return [r.content for _, r in ranked[:limit]]


async def list_memories(user_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 500))
    async with get_write_engine().connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT id, content, category, importance, updated_at FROM ai_memories "
                    "WHERE user_id = :uid ORDER BY importance DESC, updated_at DESC LIMIT :n"
                ),
                {"uid": user_id, "n": limit},
            )
        ).all()
    return [
        {
            "id": r.id,
            "content": r.content,
            "category": r.category,
            "importance": r.importance,
            "updated_at": r.updated_at,
        }
        for r in rows
    ]


async def delete_memory(user_id: str, memory_id: int) -> bool:
    async with get_write_engine().begin() as conn:
        result = await conn.execute(
            text("DELETE FROM ai_memories WHERE id = :mid AND user_id = :uid"),
            {"mid": memory_id, "uid": user_id},
        )
    return result.rowcount > 0


async def delete_all_memories(user_id: str) -> int:
    async with get_write_engine().begin() as conn:
        result = await conn.execute(
            text("DELETE FROM ai_memories WHERE user_id = :uid"), {"uid": user_id}
        )
    return result.rowcount


async def update_long_term_memory(
    user_id: str,
    session_id: str | None,
    messages: Sequence[Message],
    *,
    provider: str | None = None,
) -> int:
    """Extract and save memories from recent messages. Best-effort: never raises."""
    from app.ai.agents.memory_agent import MemoryAgent  # lazy: agents import providers

    try:
        items = await MemoryAgent(provider=provider).extract_memories(list(messages))
        return await save_memories(user_id, items, source_session=session_id) if items else 0
    except Exception:
        log.exception("Long-term memory update failed")
        return 0
