"""Question -> SQL -> guarded execution -> natural-language answer.

    generate_sql ──► run_select ──ok──► answer
         ▲              │
         └─ repair_sql ◄┘ guard rejected / Postgres error (up to SQL_MAX_REPAIRS times)

The SQL agent never executes anything itself; every attempt goes through
db/reader.run_select (sql_guard + read-only role + timeout + row cap).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.ai.agents.sql_agent import CannotAnswer, SQLAgent
from app.ai.config import settings
from app.ai.db.reader import run_select
from app.ai.db.schema_inspector import get_schema_context
from app.ai.exceptions import QueryExecutionError, QueryFailedError, SQLGuardError
from app.ai.schemas.llm_io import Message, Usage
from app.ai.utils.token_counter import truncate_to_tokens

log = logging.getLogger(__name__)

HISTORY_MESSAGES = 6        # recent turns given to the SQL agent for follow-up questions
HISTORY_MESSAGE_TOKENS = 400


@dataclass
class QAResult:
    question: str
    sql: str
    answer: str
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False
    attempts: int = 1
    explanation: str = ""
    provider: str | None = None
    model: str | None = None
    usage: Usage = field(default_factory=Usage)


def _short_history(history: list[Message] | None) -> list[Message]:
    recent = [m for m in (history or []) if m.role != "system"][-HISTORY_MESSAGES:]
    trimmed = [
        Message(role=m.role, content=truncate_to_tokens(m.content, HISTORY_MESSAGE_TOKENS))
        for m in recent
    ]
    while trimmed and trimmed[0].role != "user":
        trimmed.pop(0)
    return trimmed


async def run_qa(
    question: str,
    *,
    history: list[Message] | None = None,
    provider: str | None = None,
    max_rows: int | None = None,
    max_repairs: int | None = None,
    explain: bool = True,
) -> QAResult:
    """Raises CannotAnswer, SQLGuardError (blocked after all repairs) or QueryFailedError."""
    max_rows = max_rows or settings.SQL_MAX_ROWS
    max_repairs = settings.SQL_MAX_REPAIRS if max_repairs is None else max_repairs

    schema_text = await get_schema_context(question=question)
    if not schema_text.strip():
        raise CannotAnswer("No database tables are available to answer questions from.")

    agent = SQLAgent(provider=provider, max_rows=max_rows)
    generated = await agent.generate_sql(question, schema_text, _short_history(history))
    sql = generated.sql

    result = None
    attempt = 0
    for attempt in range(1, max_repairs + 2):
        try:
            result = await run_select(sql, max_rows=max_rows)
            break
        except (SQLGuardError, QueryExecutionError) as exc:
            log.info("SQL attempt %d failed: %s", attempt, exc)
            if attempt > max_repairs:
                if isinstance(exc, SQLGuardError):
                    raise
                raise QueryFailedError(
                    "Could not build a working query for that question."
                ) from exc
            generated = await agent.repair_sql(question, schema_text, sql, str(exc))
            sql = generated.sql
    assert result is not None

    if explain:
        answer = await agent.answer(
            question, result.sql, result.rows, truncated=result.truncated
        )
    else:
        answer = generated.explanation

    last = agent.last_response
    return QAResult(
        question=question,
        sql=result.sql,
        answer=answer,
        columns=result.columns,
        rows=result.rows,
        truncated=result.truncated,
        attempts=attempt,
        explanation=generated.explanation,
        provider=last.provider if last else None,
        model=last.model if last else None,
        usage=agent.usage,
    )
