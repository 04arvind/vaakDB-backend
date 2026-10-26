"""Answers a question from retrieved passages, with [n] citations."""
from __future__ import annotations

from typing import Sequence

from app.ai.agents.base import BaseAgent
from app.ai.db.vector_store import SearchHit
from app.ai.prompts.loader import load_prompt, render
from app.ai.schemas.llm_io import LLMResponse, Message
from app.ai.utils.token_counter import estimate_tokens, truncate_to_tokens

RAG_SYSTEM = load_prompt("rag")


def format_context(hits: Sequence[SearchHit], max_tokens: int) -> tuple[str, list[SearchHit]]:
    """Numbered passages that fit the budget. Returns (text, hits actually used)."""
    blocks: list[str] = []
    used: list[SearchHit] = []
    spent = 0
    for hit in hits:
        block = f"[{len(used) + 1}] (source: {hit.source})\n{hit.content.strip()}"
        cost = estimate_tokens(block)
        if spent + cost > max_tokens:
            room = max_tokens - spent
            if room < 150:
                break
            block, cost = truncate_to_tokens(block, room), room
        blocks.append(block)
        used.append(hit)
        spent += cost
    return "\n\n".join(blocks), used


class RAGAgent(BaseAgent):
    default_provider = "groq"
    temperature = 0.1
    max_tokens = 900

    async def answer(
        self,
        question: str,
        hits: Sequence[SearchHit],
        *,
        history: list[Message] | None = None,
        max_context_tokens: int = 3000,
    ) -> tuple[LLMResponse, list[SearchHit]]:
        context, used = format_context(hits, max_context_tokens)
        system = render(RAG_SYSTEM, CONTEXT=context or "(no passages)")
        resp = await self._ask(system, question, history=history)
        return resp, used
