"""Retrieval-augmented answering over ChromaDB.

    question ─► embed ─► search (user's documents + indexed DB rows) ─► RAGAgent ─► answer + sources

Follow-up questions ("and what about March?") are made standalone by
prepending the previous user turn to the retrieval query.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.ai.agents.rag_agent import RAGAgent
from app.ai.config import settings
from app.ai.db.vector_store import NAMESPACES, SearchHit, search
from app.ai.schemas.llm_io import Message, Usage

log = logging.getLogger(__name__)

FOLLOW_UP_MAX_WORDS = 8


@dataclass
class Source:
    index: int
    source: str
    doc_id: str
    namespace: str
    score: float
    snippet: str


@dataclass
class RAGResult:
    answer: str
    sources: list[Source] = field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    usage: Usage = field(default_factory=Usage)


def retrieval_query(question: str, history: list[Message] | None) -> str:
    """Short follow-ups are ambiguous on their own; add the previous user turn."""
    if len(question.split()) > FOLLOW_UP_MAX_WORDS or not history:
        return question
    previous = next((m.content for m in reversed(history) if m.role == "user"), "")
    return f"{previous[:500]}\n{question}" if previous else question


async def retrieve(
    question: str,
    *,
    user_id: str,
    history: list[Message] | None = None,
    k: int | None = None,
) -> list[SearchHit]:
    return await search(
        NAMESPACES,
        retrieval_query(question, history),
        user_id=user_id,
        k=k or settings.RAG_TOP_K,
        min_score=settings.RAG_MIN_SCORE,
    )


async def run_rag(
    question: str,
    *,
    user_id: str,
    history: list[Message] | None = None,
    hits: list[SearchHit] | None = None,
    provider: str | None = None,
) -> RAGResult | None:
    """Returns None when nothing relevant was retrieved (caller decides the fallback)."""
    if hits is None:
        hits = await retrieve(question, user_id=user_id, history=history)
    if not hits:
        return None

    agent = RAGAgent(provider=provider)
    recent = [m for m in (history or []) if m.role != "system"][-4:]
    while recent and recent[0].role != "user":
        recent.pop(0)
    resp, used = await agent.answer(
        question, hits, history=recent, max_context_tokens=settings.RAG_MAX_CONTEXT_TOKENS
    )
    sources = [
        Source(index=i + 1, source=h.source, doc_id=h.doc_id, namespace=h.namespace,
               score=round(h.score, 4), snippet=h.content[:240])
        for i, h in enumerate(used)
    ]
    return RAGResult(answer=resp.text.strip(), sources=sources, provider=resp.provider,
                     model=resp.model, usage=agent.usage)
