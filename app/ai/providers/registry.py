"""One shared instance per provider, built lazily from settings.

    provider = get_provider("groq")
    names = configured_providers()          # ["groq", "gemini"] if both keys are set
    await close_providers()                 # in the FastAPI shutdown handler

Tests can swap in fakes with register_provider("groq", FakeProvider()).
"""
from __future__ import annotations

import threading
from typing import Callable

from app.ai.providers.base import LLMProvider, ProviderNotConfiguredError
from app.ai.providers.gemini_provider import GeminiProvider
from app.ai.providers.groq_provider import GroqProvider


def _build_groq() -> LLMProvider:
    from app.ai.config import settings

    return GroqProvider(
        api_key=settings.api_key("groq"),
        base_url=settings.GROQ_BASE_URL,
        default_model=settings.GROQ_MODEL,
        timeout_s=settings.LLM_TIMEOUT_S,
        max_retries=settings.LLM_MAX_RETRIES,
    )


def _build_gemini() -> LLMProvider:
    from app.ai.config import settings

    return GeminiProvider(
        api_key=settings.api_key("gemini"),
        base_url=settings.GEMINI_BASE_URL,
        default_model=settings.GEMINI_MODEL,
        embed_model=settings.GEMINI_EMBED_MODEL,
        embedding_dim=settings.EMBEDDING_DIM,
        thinking_budget=settings.GEMINI_THINKING_BUDGET,
        timeout_s=settings.LLM_TIMEOUT_S,
        max_retries=settings.LLM_MAX_RETRIES,
    )


_FACTORIES: dict[str, Callable[[], LLMProvider]] = {
    "groq": _build_groq,
    "gemini": _build_gemini,
}
_instances: dict[str, LLMProvider] = {}
_lock = threading.Lock()


def known_providers() -> list[str]:
    return sorted(set(_FACTORIES) | set(_instances))


def get_provider(name: str) -> LLMProvider:
    """Return the provider instance. Raises ProviderNotConfiguredError for unknown names."""
    key = (name or "").lower()
    inst = _instances.get(key)
    if inst is not None:
        return inst
    with _lock:
        inst = _instances.get(key)
        if inst is None:
            factory = _FACTORIES.get(key)
            if factory is None:
                raise ProviderNotConfiguredError(f"Unknown provider: {name!r}.")
            inst = _instances[key] = factory()
    return inst


def is_configured(name: str) -> bool:
    try:
        return get_provider(name).is_configured
    except ProviderNotConfiguredError:
        return False


def configured_providers() -> list[str]:
    return [n for n in known_providers() if is_configured(n)]


def register_provider(name: str, provider: LLMProvider) -> None:
    """Install a custom or fake provider instance under `name`."""
    with _lock:
        _instances[name.lower()] = provider


async def close_providers() -> None:
    with _lock:
        instances = list(_instances.values())
        _instances.clear()
    for inst in instances:
        await inst.aclose()
