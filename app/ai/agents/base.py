"""Shared plumbing for all agents: provider access, prompt calls, JSON parsing.

Agents are cheap: create one per request. Each instance adds up the token usage
of its calls in `self.usage`, so workflows can report cost per request.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.ai.exceptions import AIError
from app.ai.orchestration.fallback import generate_with_fallback
from app.ai.providers.base import LLMProvider
from app.ai.providers.registry import get_provider
from app.ai.schemas.llm_io import Attachment, LLMResponse, Message, Usage

log = logging.getLogger(__name__)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class AgentError(AIError):
    """Raised when an agent cannot produce a valid result."""


class BaseAgent:
    # Subclasses override these defaults. The orchestrator can override per call.
    default_provider: str = "groq"
    temperature: float = 0.1
    max_tokens: int = 1024

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        allow_fallback: bool = True,
    ):
        self.provider_name = provider or self.default_provider
        self.model = model
        self.allow_fallback = allow_fallback
        self.usage = Usage()
        self.last_response: LLMResponse | None = None

    @property
    def provider(self) -> LLMProvider:
        return get_provider(self.provider_name)

    async def _ask(
        self,
        system: str,
        user: str,
        *,
        history: list[Message] | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        attachments: list[Attachment] | None = None,
    ) -> LLMResponse:
        messages = [Message(role="system", content=system)]
        if history:
            messages.extend(m for m in history if m.role != "system")
        messages.append(Message(role="user", content=user, attachments=attachments or []))

        resp = await generate_with_fallback(
            messages,
            preferred=self.provider_name,
            model=self.model,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.max_tokens,
            json_mode=json_mode,
            allow_fallback=self.allow_fallback,
        )
        self.usage = self.usage + resp.usage
        self.last_response = resp
        log.debug(
            "%s via %s/%s: %d tokens in %d ms",
            type(self).__name__, resp.provider, resp.model,
            resp.usage.total_tokens, resp.latency_ms,
        )
        if resp.truncated:
            log.warning("%s output hit max_tokens (%s)", type(self).__name__, resp.provider)
        return resp

    @staticmethod
    def parse_json(text: str) -> Any:
        """Parse model output as JSON, tolerating ```json fences and extra prose."""
        cleaned = _FENCE.sub("", (text or "").strip())
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            raise
