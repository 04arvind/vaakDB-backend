"""Response bodies for api/routes.py."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.ai.schemas.llm_io import Usage

Route = Literal["general", "data_query", "knowledge"]

__all__ = [
    "Route", "ChatResponse", "QueryResponse", "IngestResponse", "SessionSummary",
    "HistoryMessage", "HistoryResponse", "MemoryOut", "HealthResponse", "SourceOut",
    "DocumentOut",
]


class SourceOut(BaseModel):
    """A retrieved passage cited as [index] in a RAG answer."""

    index: int
    source: str
    doc_id: str
    namespace: str
    score: float
    snippet: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    route: Route
    provider: str | None = None
    model: str | None = None
    sql: str | None = None
    sources: list[SourceOut] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class QueryResponse(BaseModel):
    session_id: str
    answer: str
    sql: str
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] | None = None  # only when include_rows=true
    row_count: int = 0
    truncated: bool = False
    usage: Usage = Field(default_factory=Usage)


class IngestResponse(BaseModel):
    target: str
    dry_run: bool
    extracted_count: int
    inserted_count: int = 0
    inserted_ids: list[int | str] = Field(default_factory=list)
    records: list[dict[str, Any]] = Field(default_factory=list)
    source_text: str | None = Field(
        default=None, description="Text read from an uploaded file (file ingest only)."
    )


class DocumentOut(BaseModel):
    doc_id: str
    filename: str | None = None
    title: str | None = None
    mime_type: str | None = None
    chunks: int = 0
    created_at: str | None = None


class SessionSummary(BaseModel):
    session_id: str
    title: str | None = None
    updated_at: datetime


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


class MemoryOut(BaseModel):
    id: int
    content: str
    category: str
    importance: int
    updated_at: datetime


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    providers: dict[str, bool]
    database: dict[str, bool] | None = None
