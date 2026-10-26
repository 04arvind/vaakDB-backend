"""Builds the final prompt context and keeps it inside the token budget.

    ctx = build_context(
        system_prompt=SQL_SYSTEM,
        user_message=question,
        history=await get_history(session_id, user_id),
        sections=[
            ("What you know about this user", format_bullets(memories)),
            ("Relevant data", format_bullets(snippets)),
        ],
        max_tokens=6000,
    )
    await agent._ask(ctx.system, ctx.user, history=ctx.history)   # agents' _ask signature
    # or, for providers.generate(): ctx.messages()

Priority when space runs out (highest first):
    system prompt  >  current user message  >  sections (in the order given)  >  history (newest first)

Token counts are estimates from utils/token_counter.py (characters / 3.5,
deliberately pessimistic).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from app.ai.schemas.llm_io import Message
from app.ai.utils.token_counter import (
    MESSAGE_OVERHEAD as _MESSAGE_OVERHEAD,
    estimate_tokens,
    truncate_to_tokens,
)

__all__ = [
    "BuiltContext", "build_context", "estimate_tokens", "format_bullets",
    "truncate_to_tokens",
]


def format_bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {i.strip()}" for i in items if i and i.strip())


@dataclass
class BuiltContext:
    system: str
    user: str
    history: list[Message]
    token_estimate: int
    dropped_messages: int = 0
    dropped_sections: list[str] = field(default_factory=list)

    def messages(self) -> list[Message]:
        """Full list [system, *history, user] for providers that take one message list."""
        return [
            Message(role="system", content=self.system),
            *self.history,
            Message(role="user", content=self.user),
        ]


def build_context(
    *,
    system_prompt: str,
    user_message: str,
    history: Sequence[Message] = (),
    sections: Sequence[tuple[str, str]] = (),
    max_tokens: int = 6000,
    reserve_for_output: int = 1000,
    section_share: float = 0.5,
    max_message_tokens: int = 1500,
) -> BuiltContext:
    budget = max_tokens - reserve_for_output
    if budget <= 0:
        raise ValueError("reserve_for_output must be smaller than max_tokens.")

    system_tokens = estimate_tokens(system_prompt)
    if system_tokens >= budget:
        raise ValueError("The system prompt alone exceeds the token budget.")

    # The current question always goes in, but a huge paste cannot eat everything.
    user_cap = max(int((budget - system_tokens) * 0.6), 200)
    user_text = truncate_to_tokens(user_message, user_cap)
    remaining = max(budget - system_tokens - estimate_tokens(user_text), 0)

    # Sections (memories, retrieved data, ...) get at most `section_share` of what is left.
    section_budget = int(remaining * section_share)
    parts: list[str] = []
    dropped_sections: list[str] = []
    section_used = 0
    cut_off = False
    for title, body in sections:
        body = (body or "").strip()
        if not body:
            continue
        if cut_off:
            dropped_sections.append(title)
            continue
        block = f"## {title}\n{body}"
        cost = estimate_tokens(block)
        if section_used + cost <= section_budget:
            parts.append(block)
            section_used += cost
            continue
        room = section_budget - section_used
        if room >= 100:  # worth keeping a truncated version
            parts.append(truncate_to_tokens(block, room))
            section_used += room
        dropped_sections.append(title)
        cut_off = True  # lower-priority sections after this one are dropped

    system = system_prompt + ("\n\n" + "\n\n".join(parts) if parts else "")

    # History: newest messages first, as many as fit, kept contiguous.
    history_budget = remaining - section_used
    chat = [m for m in history if m.role != "system"]
    kept: list[Message] = []
    history_used = 0
    for msg in reversed(chat):
        content = truncate_to_tokens(msg.content, max_message_tokens)
        cost = estimate_tokens(content) + _MESSAGE_OVERHEAD
        if history_used + cost > history_budget:
            break
        kept.append(Message(role=msg.role, content=content))
        history_used += cost
    kept.reverse()
    while kept and kept[0].role != "user":  # providers expect a conversation to start with the user
        history_used -= estimate_tokens(kept[0].content) + _MESSAGE_OVERHEAD
        kept.pop(0)

    return BuiltContext(
        system=system,
        user=user_text,
        history=kept,
        token_estimate=system_tokens + section_used + estimate_tokens(user_text) + history_used,
        dropped_messages=len(chat) - len(kept),
        dropped_sections=dropped_sections,
    )