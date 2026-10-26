"""Conversation history, long-term memory and prompt-context building."""
from app.ai.memory.context_builder import (
    BuiltContext,
    build_context,
    estimate_tokens,
    format_bullets,
)
from app.ai.memory.conversation import (
    add_exchange,
    add_message,
    delete_session,
    ensure_session,
    get_history,
    get_history_records,
    list_sessions,
)
from app.ai.memory.long_term import (
    delete_all_memories,
    delete_memory,
    list_memories,
    recall,
    save_memories,
    update_long_term_memory,
)
from app.ai.memory.tables import ensure_memory_schema

__all__ = [
    "ensure_memory_schema",
    "ensure_session", "add_exchange", "add_message", "get_history",
    "get_history_records", "list_sessions", "delete_session",
    "save_memories", "recall", "list_memories", "delete_memory",
    "delete_all_memories", "update_long_term_memory",
    "build_context", "BuiltContext", "format_bullets", "estimate_tokens",
]
