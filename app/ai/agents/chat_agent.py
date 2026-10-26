"""Plain conversational replies (no database access)."""
from __future__ import annotations

from app.ai.agents.base import BaseAgent
from app.ai.prompts.loader import load_prompt
from app.ai.schemas.llm_io import LLMResponse, Message

CHAT_SYSTEM = load_prompt("chat")


class ChatAgent(BaseAgent):
    default_provider = "groq"
    temperature = 0.5
    max_tokens = 1024

    async def reply(
        self,
        message: str,
        *,
        history: list[Message] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """`system` lets callers pass a context-enriched prompt (see memory/context_builder)."""
        return await self._ask(system or CHAT_SYSTEM, message, history=history)
