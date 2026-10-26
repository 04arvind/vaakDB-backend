"""Google Gemini (Generative Language REST API). Used for long context,
multimodal input (images, PDFs, audio) and embeddings."""
from __future__ import annotations

import base64
from typing import Any

from app.ai.providers.base import ContentBlockedError, LLMProvider, ProviderError
from app.ai.schemas.llm_io import LLMResponse, Message, Usage

_EMBED_BATCH = 100  # API maximum per batchEmbedContents call


class GeminiProvider(LLMProvider):
    name = "gemini"
    supports_attachments = True
    supports_embeddings = True

    def __init__(
        self,
        *,
        embed_model: str,
        embedding_dim: int | None = None,
        thinking_budget: int | None = 0,
        **kw: Any,
    ):
        super().__init__(**kw)
        self.embed_model = embed_model
        self.embedding_dim = embedding_dim
        self.thinking_budget = thinking_budget

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self._api_key or ""}

    async def generate(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResponse:
        model = model or self.default_model
        system = "\n\n".join(m.content for m in messages if m.role == "system")

        contents: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            parts: list[dict[str, Any]] = [
                {"inlineData": {"mimeType": a.mime_type, "data": base64.b64encode(a.data).decode()}}
                for a in m.attachments
            ]
            if m.content:
                parts.append({"text": m.content})
            role = "model" if m.role == "assistant" else "user"
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)  # Gemini wants alternating roles
            else:
                contents.append({"role": role, "parts": parts})

        config: dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if json_mode:
            config["responseMimeType"] = "application/json"
        if self.thinking_budget is not None:
            config["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}

        payload: dict[str, Any] = {"contents": contents, "generationConfig": config}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        started = self._now_ms()
        data = await self._post_json(f"/models/{model}:generateContent", payload)

        blocked = (data.get("promptFeedback") or {}).get("blockReason")
        if blocked:
            raise ContentBlockedError(f"Prompt blocked: {blocked}", provider=self.name)

        candidates = data.get("candidates") or []
        if not candidates:
            raise ProviderError("No candidates returned.", provider=self.name)
        first = candidates[0]
        finish = first.get("finishReason")
        parts = (first.get("content") or {}).get("parts") or []
        # Skip "thought" parts some models return alongside the answer.
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text and finish in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"):
            raise ContentBlockedError(f"Response blocked: {finish}", provider=self.name)

        usage = data.get("usageMetadata") or {}
        return LLMResponse(
            text=text,
            provider=self.name,
            model=data.get("modelVersion") or model,
            finish_reason=finish,
            latency_ms=int(self._now_ms() - started),
            usage=Usage(
                prompt_tokens=usage.get("promptTokenCount", 0),
                completion_tokens=usage.get("candidatesTokenCount", 0),
                total_tokens=usage.get("totalTokenCount", 0),
            ),
        )

    async def embed(
        self, texts: list[str], *, task_type: str | None = None
    ) -> list[list[float]]:
        if not texts:
            return []
        model_ref = f"models/{self.embed_model}"
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = texts[start : start + _EMBED_BATCH]
            requests = []
            for t in batch:
                req: dict[str, Any] = {"model": model_ref, "content": {"parts": [{"text": t}]}}
                if self.embedding_dim:
                    req["outputDimensionality"] = self.embedding_dim
                if task_type:
                    req["taskType"] = task_type
                requests.append(req)
            data = await self._post_json(
                f"/{model_ref}:batchEmbedContents", {"requests": requests}
            )
            try:
                vectors.extend(e["values"] for e in data["embeddings"])
            except (KeyError, TypeError) as exc:
                raise ProviderError("Unexpected embedding response.", provider=self.name) from exc
        if len(vectors) != len(texts):
            raise ProviderError("Embedding count mismatch.", provider=self.name)
        return vectors

