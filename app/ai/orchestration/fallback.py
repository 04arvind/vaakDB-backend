"""Provider fallback: try the preferred provider, then the others in
PROVIDER_FALLBACK_ORDER, skipping any that are not configured.

Falls through on rate limits, outages, timeouts and bad keys. Does NOT fall
through on other 4xx errors or safety blocks: those mean the request itself is
the problem and another provider is unlikely to help.

A `model` override only applies to the preferred provider; fallbacks use their
own default model.
"""
from __future__ import annotations

import logging

from app.ai.providers.base import (
    ProviderAuthError,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from app.ai.providers.registry import get_provider, is_configured
from app.ai.schemas.llm_io import LLMResponse, Message

log = logging.getLogger(__name__)

_FALL_THROUGH = (
    ProviderRateLimitError,
    ProviderUnavailableError,
    ProviderAuthError,
    ProviderNotConfiguredError,
)


def provider_chain(preferred: str | None, *, needs_attachments: bool = False) -> list[str]:
    """Ordered, de-duplicated list of configured providers to try."""
    from app.ai.config import settings

    order = [preferred] if preferred else []
    order += [p for p in settings.PROVIDER_FALLBACK_ORDER if p != preferred]
    chain: list[str] = []
    for name in order:
        if name in chain or not is_configured(name):
            continue
        if needs_attachments and not get_provider(name).supports_attachments:
            continue
        chain.append(name)
    return chain


async def generate_with_fallback(
    messages: list[Message],
    *,
    preferred: str | None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    json_mode: bool = False,
    allow_fallback: bool = True,
) -> LLMResponse:
    needs_attachments = any(m.attachments for m in messages)
    chain = provider_chain(preferred, needs_attachments=needs_attachments)
    if not allow_fallback:
        chain = chain[:1] if chain and chain[0] == preferred else []
    if not chain:
        wanted = preferred or "any"
        raise ProviderNotConfiguredError(
            f"No configured provider available (wanted: {wanted}"
            f"{', with attachment support' if needs_attachments else ''})."
        )

    last_error: ProviderError | None = None
    for name in chain:
        try:
            return await get_provider(name).generate(
                messages,
                model=model if name == preferred else None,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        except _FALL_THROUGH as exc:
            log.warning("Provider %s failed (%s); trying next provider", name, exc)
            last_error = exc
    assert last_error is not None
    raise last_error
