"""Backward-compatible alias. Providers live in app.ai.providers."""
from app.ai.providers import (  # noqa: F401
    LLMProvider,
    ProviderError,
    configured_providers,
    get_provider,
)
