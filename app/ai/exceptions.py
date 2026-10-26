"""Exception hierarchy for the AI module.

Everything raised on purpose by app.ai derives from AIError, so callers (and
api/routes.py) can map failures to HTTP codes without catching bare Exception.
Messages of ValueError subclasses are written to be safe to show to end users.
"""
from __future__ import annotations


class AIError(Exception):
    """Base class for all AI-module errors."""


class ConfigurationError(AIError, RuntimeError):
    """A required setting (database URL, ...) is missing."""


# --------------------------------------------------------------------------- #
# SQL / database
# --------------------------------------------------------------------------- #
class SQLGuardError(AIError, ValueError):
    """LLM-written SQL was rejected by the safety guard."""


class QueryExecutionError(AIError):
    """Postgres rejected or failed the query. The message is safe to feed back to
    the SQL agent (repair_sql) but should NOT be shown to end users."""


class QueryFailedError(AIError):
    """The SQL agent could not produce a working query, even after repairs."""


class UnknownTargetError(AIError, ValueError):
    """Ingest target is not registered in db/writer.py."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Unknown ingest target: {name!r}.")


# --------------------------------------------------------------------------- #
# Conversations / users
# --------------------------------------------------------------------------- #
class SessionNotFoundError(AIError, ValueError):
    """Session does not exist or belongs to another user (deliberately the same error)."""

    def __init__(self, message: str = "Session not found."):
        super().__init__(message)


class RateLimitExceeded(AIError):
    """The caller sent too many requests."""

    def __init__(self, retry_after: float):
        self.retry_after = max(retry_after, 0.0)
        super().__init__(f"Rate limit exceeded. Retry after {self.retry_after:.1f}s.")
