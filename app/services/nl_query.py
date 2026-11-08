"""Natural-language SELECT inside the caller's own workspace.

Reuses the AI module's SQL agent (generation, repair, answer) and SQL guard
unchanged, but scopes everything to one schema of the caller's workspace:

1. the agent only sees the tables of that schema (logical names, unqualified)
2. app.ai.db.sql_guard.validate_sql() accepts only SELECTs over exactly those
   tables (no other schema, no system catalogs, no dangerous functions, LIMIT)
3. every table reference is then rewritten to the caller's PHYSICAL schema
   (u<id>_<name>), so the query can only ever read the caller's own data
4. it runs on the read-only engine in a READ ONLY transaction
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlglot import exp

from app.ai.db.sql_guard import SENSITIVE_COLUMN_RE, SQLGuardError, validate_sql
from app.core.config import settings
from app.core.exceptions import AppError, ErrorCode
from app.db.target_db import get_read_engine
from app.security.authorization import Caller, schema_not_found, to_physical
from app.services import schema_service
from app.services.schema_service import CatalogTable
from app.sql.validator import to_jsonable

log = logging.getLogger(__name__)

MAX_ROWS = 100
MAX_REPAIRS = 2


@dataclass
class WorkspaceQueryResult:
    sql: str  # the query as shown to the user (logical names)
    answer: str
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False


def _schema_text(tables: list[CatalogTable]) -> str:
    lines: list[str] = []
    for t in tables:
        lines.append(f"TABLE {t.name}")
        for c in t.columns:
            if SENSITIVE_COLUMN_RE.search(c.name):
                continue
            flags = [f for f, on in (("PK", c.primary_key), ("nullable", c.nullable)) if on]
            lines.append(f"  - {c.name}: {c.display_type}" + (f" [{', '.join(flags)}]" if flags else ""))
    return "\n".join(lines)


def _load_tables(caller: Caller, schema: str) -> tuple[str, list[CatalogTable]]:
    with get_read_engine().connect() as conn:
        physical = to_physical(caller, schema)
        if not schema_service.schema_exists(conn, physical):
            raise schema_not_found(schema)
        tables = []
        for t in schema_service.list_tables(conn, physical):
            info = schema_service.get_table(conn, physical, t.name)
            if info is not None:
                tables.append(info)
    return physical, tables


def scope_to_workspace(sql: str, *, logical: str, physical: str, allowed: set[str]) -> str:
    """Rewrite every real table reference to physical.table. The SQL has already
    passed validate_sql(), which only allows tables in `allowed`, unqualified or
    qualified with the logical schema."""
    tree = sqlglot.parse_one(sql, read="postgres")
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    for tbl in tree.find_all(exp.Table):
        name = tbl.name.lower()
        db = (tbl.db or "").lower()
        if not db and name in ctes:
            continue  # reference to a CTE of this query
        if name not in allowed or db not in ("", logical):  # validate_sql should prevent this
            raise SQLGuardError(f"Table not allowed: {name}.")
        tbl.set("db", exp.to_identifier(physical))
        tbl.set("catalog", None)
    return tree.sql(dialect="postgres")


def _execute(sql: str) -> tuple[list[str], list[dict[str, Any]]]:
    with get_read_engine().connect() as conn, conn.begin():
        conn.execute(text("SET TRANSACTION READ ONLY"))
        result = conn.exec_driver_sql(sql)
        columns = list(result.keys())
        rows = [dict(r) for r in result.mappings().fetchmany(MAX_ROWS + 1)]
    return columns, rows


async def run_workspace_query(caller: Caller, question: str, schema: str) -> WorkspaceQueryResult:
    """Raises AppError (QUERY_FAILED / SCHEMA_NOT_FOUND / LLM_UNAVAILABLE ...)."""
    from app.ai.agents.base import AgentError
    from app.ai.agents.sql_agent import CannotAnswer, SQLAgent
    from app.ai.providers.base import ProviderError

    physical, tables = await asyncio.to_thread(_load_tables, caller, schema)
    if not tables:
        raise AppError(ErrorCode.QUERY_FAILED, f"Schema {schema} has no tables to query yet.")
    allowed = {t.name.lower() for t in tables}
    schema_text = _schema_text(tables)
    agent = SQLAgent(provider=settings.llm_provider, max_rows=MAX_ROWS)

    try:
        generated = await agent.generate_sql(question, schema_text)
        sql = generated.sql
        safe_sql = ""
        columns: list[str] = []
        fetched: list[dict[str, Any]] = []
        for attempt in range(MAX_REPAIRS + 1):
            try:
                safe_sql = validate_sql(sql, allowed, max_rows=MAX_ROWS, schema=schema)
                scoped = scope_to_workspace(safe_sql, logical=schema, physical=physical,
                                            allowed=allowed)
                columns, fetched = await asyncio.to_thread(_execute, scoped)
                break
            except (SQLGuardError, DBAPIError) as exc:
                reason = (str(exc.orig).splitlines()[0][:300]
                          if isinstance(exc, DBAPIError) and exc.orig is not None else str(exc))
                log.info("Workspace query attempt %d failed: %s", attempt + 1, reason)
                if attempt == MAX_REPAIRS:
                    raise AppError(ErrorCode.QUERY_FAILED,
                                   "Could not build a working query for that question.") from None
                sql = (await agent.repair_sql(question, schema_text, sql, reason)).sql

        truncated = len(fetched) > MAX_ROWS
        keep = [c for c in columns if not SENSITIVE_COLUMN_RE.search(c)]
        rows = [{c: to_jsonable(r[c]) for c in keep} for r in fetched[:MAX_ROWS]]
        answer = await agent.answer(question, safe_sql, rows, truncated=truncated)
    except CannotAnswer as exc:
        raise AppError(ErrorCode.QUERY_FAILED,
                       str(exc)[:300] or "I can't answer that from your data.") from None
    except (ProviderError, AgentError):
        raise AppError(ErrorCode.LLM_UNAVAILABLE) from None

    return WorkspaceQueryResult(sql=safe_sql, answer=answer, columns=keep, rows=rows,
                                truncated=truncated)
