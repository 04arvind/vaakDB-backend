"""Backward-compatible alias. The implementation lives in app.ai.memory.context_builder."""
from app.ai.memory.context_builder import (  # noqa: F401
    BuiltContext,
    build_context,
    estimate_tokens,
    format_bullets,
    truncate_to_tokens,
)
