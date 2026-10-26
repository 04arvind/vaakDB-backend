"""Groq (OpenAI-compatible chat completions API). Fast and cheap: the default
for SQL generation, routing, extraction and memory."""
from __future__ import annotations

from typing import Any

from app.ai.providers.base import LLMProvider, ProviderError
from app.ai.schemas.llm_io import LLMResponse, Message, Usage


class GroqProvider(LLMProvider):
    name = "groq"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def generate(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResponse:
        if any(m.attachments for m in messages):
            raise ProviderError("Attachments are not supported.", provider=self.name)

        model = model or self.default_model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            # Groq requires the word "JSON" in the prompt for json_object mode;
            # every json_mode prompt in this module asks for JSON explicitly.
            payload["response_format"] = {"type": "json_object"}

        started = self._now_ms()
        data = await self._post_json("/chat/completions", payload)

        try:
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Unexpected response shape.", provider=self.name) from exc

        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            provider=self.name,
            model=data.get("model") or model,
            finish_reason=choice.get("finish_reason"),
            latency_ms=int(self._now_ms() - started),
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )
