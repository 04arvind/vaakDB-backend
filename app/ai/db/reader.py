"""Runs validated, read-only queries.

Pipeline: schema allow-list -> sql_guard -> READ ONLY transaction with timeout
-> row cap -> sensitive columns stripped from the result -> JSON-safe values.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.exc import DBAPIError

from app.ai.db.engine import cfg, get_read_engine
from app.ai.db.schema_inspector import get_allowed_tables, get_hidden_tables
from app.ai.db.sql_guard import SENSITIVE_COLUMN_RE, validate_sql
from app.ai.exceptions import QueryExecutionError

log = logging.getLogger(__name__)

__all__ = ["QueryExecutionError", "QueryResult", "run_select"]

MAX_CELL_CHARS = 2000


@dataclass
class QueryResult:
    sql: str  # the guarded SQL that actually ran (its LIMIT may be max_rows + 1)
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False

    @property
    def row_count(self) -> int:
        return len(self.rows)


def _jsonable(value: Any) -> Any:
    """Keep results safe to serialize and to put in a prompt."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<binary {len(bytes(value))} bytes>"
    if isinstance(value, str) and len(value) > MAX_CELL_CHARS:
        return value[:MAX_CELL_CHARS] + "...[truncated]"
    return value


async def run_select(sql: str, *, max_rows: int | None = None) -> QueryResult:
    """Validate and execute LLM-written SQL. Raises SQLGuardError or QueryExecutionError."""
    max_rows = max_rows or int(cfg("SQL_MAX_ROWS", 100))
    timeout_ms = int(cfg("SQL_TIMEOUT_MS", 5000))

    safe_sql = validate_sql(
        sql,
        await get_allowed_tables(),
        blocked_tables=await get_hidden_tables(),
        max_rows=max_rows + 1,  # one extra row, so we can tell the result was cut off
        schema=cfg("DB_SCHEMA", "public"),
    )

    try:
        async with get_read_engine().begin() as conn:
            await conn.exec_driver_sql("SET TRANSACTION READ ONLY")
            await conn.exec_driver_sql(f"SET LOCAL statement_timeout = {timeout_ms}")
            result = await conn.exec_driver_sql(safe_sql)
            columns = list(result.keys())
            fetched = [dict(r) for r in result.mappings().fetchmany(max_rows + 1)]
    except DBAPIError as exc:
        orig = str(exc.orig).strip() if exc.orig is not None else ""
        message = orig.splitlines()[0][:300] if orig else "query failed"
        log.warning("Read query failed: %s | sql=%s", message, safe_sql)
        raise QueryExecutionError(message) from exc

    truncated = len(fetched) > max_rows
    fetched = fetched[:max_rows]

    # Last line of defence: `SELECT *` must never leak sensitive columns.
    keep = [c for c in columns if not SENSITIVE_COLUMN_RE.search(c)]
    rows = [{c: _jsonable(row[c]) for c in keep} for row in fetched]

    log.info("Read query ok: %d rows (truncated=%s)", len(rows), truncated)
    return QueryResult(sql=safe_sql, columns=keep, rows=rows, truncated=truncated)
