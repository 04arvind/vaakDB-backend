"""Turns workflow results into API response objects (one place for the mapping)."""
from __future__ import annotations

from typing import Any

from app.ai.schemas.llm_io import Usage
from app.ai.schemas.responses import ChatResponse, IngestResponse, QueryResponse
from app.ai.workflows.ingest_workflow import IngestResult
from app.ai.workflows.qa_workflow import QAResult


def query_response(session_id: str, qa: QAResult, *, include_rows: bool) -> QueryResponse:
    return QueryResponse(
        session_id=session_id,
        answer=qa.answer,
        sql=qa.sql,
        columns=qa.columns,
        rows=qa.rows if include_rows else None,
        row_count=len(qa.rows),
        truncated=qa.truncated,
        usage=qa.usage,
    )


def chat_response(
    session_id: str,
    answer: str,
    *,
    route: str,
    provider: str | None,
    model: str | None,
    usage: Usage | None,
    sql: str | None = None,
    sources: list[dict[str, Any]] | None = None,
) -> ChatResponse:
    return ChatResponse(
        session_id=session_id,
        answer=answer,
        route=route,  # type: ignore[arg-type]
        provider=provider,
        model=model,
        sql=sql,
        sources=sources or [],
        usage=usage or Usage(),
    )


def ingest_response(result: IngestResult, *, source_text: str | None = None) -> IngestResponse:
    return IngestResponse(
        target=result.target,
        dry_run=result.dry_run,
        extracted_count=len(result.records),
        inserted_count=len(result.inserted_ids),
        inserted_ids=result.inserted_ids,
        records=result.records,
        source_text=source_text,
    )


def exchange_meta(*, route: str, provider: str | None, model: str | None,
                  usage: Usage, sql: str | None = None) -> dict:
    """Metadata stored with the assistant message in ai_messages.meta."""
    meta = {"route": route, "provider": provider, "model": model,
            "tokens": usage.total_tokens}
    if sql:
        meta["sql"] = sql
    return meta
