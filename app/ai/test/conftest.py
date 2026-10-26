"""Shared fixtures. Tests run without network, database or vector store:
providers are replaced by FakeProvider and DB/Chroma calls are monkeypatched.

Run from the repository root:
    pytest app/ai/test
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ai.providers import registry  # noqa: E402
from app.ai.providers.base import LLMProvider  # noqa: E402
from app.ai.schemas.llm_io import LLMResponse, Message, Usage  # noqa: E402

Reply = str | Callable[[list[Message], bool], str] | Exception


class FakeProvider(LLMProvider):
    """Returns scripted replies in order and records every call."""

    supports_attachments = True
    supports_embeddings = True

    def __init__(self, name: str, replies: list[Reply] | None = None):
        super().__init__(api_key="test", base_url="http://fake", default_model=f"{name}-model")
        self.name = name
        self.replies: list[Reply] = list(replies or [])
        self.calls: list[dict] = []

    async def generate(self, messages, *, model=None, temperature=0.2, max_tokens=1024,
                       json_mode=False) -> LLMResponse:
        self.calls.append({"messages": messages, "json_mode": json_mode, "model": model})
        if not self.replies:
            raise AssertionError(f"FakeProvider {self.name}: no scripted reply left")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        text = reply(messages, json_mode) if callable(reply) else reply
        return LLMResponse(text=text, provider=self.name, model=model or self.default_model,
                           usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    async def embed(self, texts, *, task_type=None):
        """Deterministic vectors of the configured size (real ones come from Gemini)."""
        from app.ai.config import settings

        dim = settings.EMBEDDING_DIM
        return [[((hash(t) >> (i % 32)) & 7) / 7.0 + 0.01 for i in range(dim)] for t in texts]


@pytest.fixture
def fake_providers():
    """Install fake groq + gemini providers; restore the real registry afterwards."""
    saved = dict(registry._instances)
    groq, gemini = FakeProvider("groq"), FakeProvider("gemini")
    registry.register_provider("groq", groq)
    registry.register_provider("gemini", gemini)
    try:
        yield groq, gemini
    finally:
        registry._instances.clear()
        registry._instances.update(saved)
