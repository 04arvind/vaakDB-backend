from __future__ import annotations

from pydantic import BaseModel


class EngineStatus(BaseModel):
    ok: bool
    role: str | None = None
    database: str | None = None
    superuser: bool | None = None
    error: str | None = None


class ConnectionStatus(BaseModel):
    """Status of the configured databases. URLs and passwords are never returned."""

    meta_db: bool
    target_read: EngineStatus
    target_write: EngineStatus
    default_schema: str
    ai_routes_enabled: bool
    warnings: list[str] = []
