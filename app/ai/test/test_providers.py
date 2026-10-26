"""Providers against a mocked HTTP layer (httpx.MockTransport): wire format,
error mapping, retries and cross-provider fallback."""
from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest

from app.ai.orchestration.fallback import generate_with_fallback
from app.ai.providers import registry
from app.ai.providers.base import (
    ProviderAuthError,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderRateLimitError,
)
from app.ai.providers.gemini_provider import GeminiProvider
from app.ai.providers.groq_provider import GroqProvider
from app.ai.schemas.llm_io import Attachment, Message

MSGS = [Message(role="system", content="sys"), Message(role="user", content="hi")]


def run(coro):
    return asyncio.run(coro)


def groq(handler, **kw) -> GroqProvider:
    return GroqProvider(api_key="k", base_url="https://groq.test/v1", default_model="llama",
                        transport=httpx.MockTransport(handler), **kw)


def gemini(handler, **kw) -> GeminiProvider:
    return GeminiProvider(api_key="g", base_url="https://gemini.test/v1beta",
                          default_model="gemini-x", embed_model="emb", embedding_dim=3,
                          transport=httpx.MockTransport(handler), **kw)


GROQ_OK = {
    "model": "llama", "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}


# ---- Groq ------------------------------------------------------------------ #
def test_groq_request_and_response_shape():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        seen["path"] = req.url.path
        return httpx.Response(200, json=GROQ_OK)

    resp = run(groq(handler).generate(MSGS, json_mode=True, max_tokens=50))
    assert resp.text == "hello" and resp.provider == "groq"
    assert resp.usage.total_tokens == 4
    assert seen["auth"] == "Bearer k"
    assert seen["path"].endswith("/chat/completions")
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["body"]["max_tokens"] == 50


def test_groq_retries_rate_limit_then_succeeds():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0.01"},
                                  json={"error": {"message": "slow down"}})
        return httpx.Response(200, json=GROQ_OK)

    assert run(groq(handler, max_retries=2).generate(MSGS)).text == "hello"
    assert calls["n"] == 2


def test_groq_rate_limit_exhausted_raises():
    handler = lambda req: httpx.Response(429, headers={"retry-after": "0.01"}, json={})
    with pytest.raises(ProviderRateLimitError):
        run(groq(handler, max_retries=1).generate(MSGS))


def test_groq_auth_error_is_not_retried():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with pytest.raises(ProviderAuthError):
        run(groq(handler, max_retries=3).generate(MSGS))
    assert calls["n"] == 1


def test_groq_bad_request_maps_to_provider_error():
    handler = lambda req: httpx.Response(400, json={"error": {"message": "nope"}})
    with pytest.raises(ProviderError) as info:
        run(groq(handler).generate(MSGS))
    assert info.value.status_code == 400


def test_groq_rejects_attachments():
    msgs = [Message(role="user", content="x",
                    attachments=[Attachment(mime_type="image/png", data=b"1")])]
    with pytest.raises(ProviderError):
        run(groq(lambda r: httpx.Response(200, json=GROQ_OK)).generate(msgs))


def test_missing_key_raises_not_configured():
    p = GroqProvider(api_key=None, base_url="https://x", default_model="m")
    assert not p.is_configured
    with pytest.raises(ProviderNotConfiguredError):
        run(p.generate(MSGS))


# ---- Gemini ---------------------------------------------------------------- #
def test_gemini_request_shape_and_multimodal():
    seen = {}

    def handler(req):
        seen["key"] = req.headers["x-goog-api-key"]
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "thinking", "thought": True},
                                                  {"text": "answer"}]},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2,
                              "totalTokenCount": 9},
        })

    msgs = [
        Message(role="system", content="be brief"),
        Message(role="user", content="earlier"),
        Message(role="assistant", content="ok"),
        Message(role="user", content="read this",
                attachments=[Attachment(mime_type="image/png", data=b"\x89PNG")]),
    ]
    resp = run(gemini(handler).generate(msgs, json_mode=True))
    assert resp.text == "answer" and resp.usage.total_tokens == 9
    body = seen["body"]
    assert seen["key"] == "g" and seen["path"].endswith("/models/gemini-x:generateContent")
    assert body["systemInstruction"]["parts"][0]["text"] == "be brief"
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    inline = body["contents"][2]["parts"][0]["inlineData"]
    assert inline["mimeType"] == "image/png"
    assert base64.b64decode(inline["data"]) == b"\x89PNG"
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


def test_gemini_blocked_prompt():
    handler = lambda r: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(ProviderError):
        run(gemini(handler).generate(MSGS))


def test_gemini_embed_batches_and_task_type():
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(body)
        return httpx.Response(200, json={"embeddings": [{"values": [0.1, 0.2, 0.3]}
                                                        for _ in body["requests"]]})

    vectors = run(gemini(handler).embed(["a"] * 150, task_type="RETRIEVAL_QUERY"))
    assert len(vectors) == 150 and len(seen) == 2  # 100 + 50
    first = seen[0]["requests"][0]
    assert first["taskType"] == "RETRIEVAL_QUERY" and first["outputDimensionality"] == 3


# ---- fallback -------------------------------------------------------------- #
def test_fallback_moves_to_next_provider(fake_providers):
    groq_fake, gemini_fake = fake_providers
    groq_fake.replies = [ProviderRateLimitError("busy", provider="groq")]
    gemini_fake.replies = ["from gemini"]
    resp = run(generate_with_fallback(MSGS, preferred="groq"))
    assert resp.provider == "gemini" and resp.text == "from gemini"


def test_fallback_does_not_hide_bad_requests(fake_providers):
    groq_fake, gemini_fake = fake_providers
    groq_fake.replies = [ProviderError("bad request", provider="groq", status_code=400)]
    gemini_fake.replies = ["unused"]
    with pytest.raises(ProviderError):
        run(generate_with_fallback(MSGS, preferred="groq"))
    assert gemini_fake.replies == ["unused"]


def test_attachments_only_go_to_multimodal_providers(fake_providers):
    groq_fake, gemini_fake = fake_providers
    groq_fake.supports_attachments = False
    gemini_fake.replies = ["seen"]
    msgs = [Message(role="user", content="x",
                    attachments=[Attachment(mime_type="image/png", data=b"1")])]
    assert run(generate_with_fallback(msgs, preferred="groq")).provider == "gemini"
    assert groq_fake.calls == []


def test_registry_unknown_provider():
    with pytest.raises(ProviderNotConfiguredError):
        registry.get_provider("nope")
