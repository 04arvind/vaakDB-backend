"""Structured LLM output for the chat -> DB operations flow.

    model = get_chat_model()
    decision = await model.with_structured_output(IntentDecision).ainvoke(
        [("system", ROUTER_SYSTEM), ("user", message)]
    )

Same shape as LangChain's `.with_structured_output()`, built on the module's
existing Groq/Gemini providers (and their provider fallback), so no extra SDKs
are needed:
- temperature 0, JSON mode
- the Pydantic model's JSON schema is put in the system prompt and the reply
  is validated against it; one corrective re-ask on invalid output, then
  LLMBadOutputError
- rate limits / timeouts / outages: one retry with backoff, then LLMUnavailableError
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Sequence
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from app.ai.agents.base import AgentError, BaseAgent
from app.ai.exceptions import AIError, ConfigurationError
from app.ai.providers.base import (
    ContentBlockedError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
T_co = TypeVar("T_co", bound=BaseModel, covariant=True)
MessageLike = tuple[str, str]  # (role, content), role in system|user|assistant

RETRY_BACKOFF_S = 2.0
MAX_BACKOFF_S = 8.0


class LLMUnavailableError(AIError):
    """Provider down, rate-limited, timing out or not configured (after one retry)."""


class LLMBadOutputError(AIError):
    """The model's answer did not match the required structure (after one re-ask)."""


class StructuredRunnable(Protocol[T_co]):  # what graph nodes depend on (fakes in tests)
    async def ainvoke(self, messages: Sequence[MessageLike]) -> T_co: ...


class ChatModel(Protocol):
    def with_structured_output(self, schema: type[T]) -> StructuredRunnable[T]: ...


class _Agent(BaseAgent):
    temperature = 0.0
    max_tokens = 4096


def _split_messages(messages: Sequence[MessageLike]) -> tuple[str, str]:
    system = "\n\n".join(c for r, c in messages if r == "system")
    user = "\n\n".join(c for r, c in messages if r != "system")
    return system, user


def _schema_instructions(schema: type[BaseModel]) -> str:
    return (
        "Reply with ONE JSON object and nothing else (no prose, no markdown). It must validate "
        "against this JSON Schema:\n" + json.dumps(schema.model_json_schema(by_alias=True))
    )


class _Structured(Generic[T]):
    def __init__(self, schema: type[T], provider: str):
        self.schema = schema
        self.provider = provider

    async def _ask(self, agent: _Agent, system: str, user: str) -> str:
        for attempt in (1, 2):
            try:
                resp = await agent._ask(system, user, json_mode=True, temperature=0.0)
                return resp.text
            except ContentBlockedError as exc:
                raise LLMBadOutputError("The request was blocked by the AI provider.") from exc
            except (ProviderRateLimitError, ProviderUnavailableError) as exc:
                if attempt == 2:
                    raise LLMUnavailableError(str(exc)) from exc
                wait = getattr(exc, "retry_after", None) or RETRY_BACKOFF_S
                log.info("LLM busy (%s); retrying in %.1fs", type(exc).__name__, wait)
                await asyncio.sleep(min(float(wait), MAX_BACKOFF_S))
            except (ProviderError, ConfigurationError, AgentError) as exc:
                raise LLMUnavailableError(str(exc)) from exc
        raise LLMUnavailableError("unreachable")  # pragma: no cover

    async def ainvoke(self, messages: Sequence[MessageLike]) -> T:
        system, user = _split_messages(messages)
        system = f"{system}\n\n{_schema_instructions(self.schema)}"
        agent = _Agent(provider=self.provider)
        prompt = user
        last_error = ""
        for _ in range(2):
            text = await self._ask(agent, system, prompt)
            try:
                data: Any = BaseAgent.parse_json(text)
                if isinstance(data, list) and len(data) == 1:
                    data = data[0]
                return self.schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                last_error = str(exc)[:800]
                log.info("Structured output invalid for %s: %s", self.schema.__name__, last_error[:200])
                prompt = (f"{user}\n\nYour previous reply was invalid:\n{last_error}\n"
                          "Return a corrected JSON object only.")
        raise LLMBadOutputError(last_error or "invalid output")


class ProviderChatModel:
    def __init__(self, provider: str):
        self.provider = provider

    def with_structured_output(self, schema: type[T]) -> StructuredRunnable[T]:
        return _Structured(schema, self.provider)


def get_chat_model(provider: str | None = None) -> ChatModel:
    """Provider from LLM_PROVIDER (groq | gemini); the module's fallback order
    still applies if that provider fails."""
    name = (provider or os.getenv("LLM_PROVIDER") or "groq").strip().lower()
    if name not in ("groq", "gemini"):
        raise ConfigurationError(f"Unknown LLM_PROVIDER {name!r} (use groq or gemini).")
    return ProviderChatModel(name)
