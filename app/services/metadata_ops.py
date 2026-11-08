"""Metadata operations (no LLM): list schemas, list tables, describe a table,
paginated rows, CSV export, all inside the caller's own workspace.

Users pass logical schema names; every catalog lookup goes through
to_physical(caller, ...), so another user's schema can't even be named. A table
that belongs to someone else gives the same TABLE_NOT_FOUND as one that does
not exist.
"""
from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from typing import Any, Literal

from sqlalchemy import Connection, asc, column, desc, func, literal_column, select, table, text

from app.ai.db.sql_guard import SENSITIVE_COLUMN_RE
from app.core.config import settings
from app.core.exceptions import AppError, ErrorCode
from app.db.target_db import get_read_engine
from app.security.authorization import (
    Caller,
    schema_not_found,
    table_not_found,
    to_logical,
    to_physical,
)
from app.services import schema_service
from app.services.schema_service import CatalogTable
from app.sql.parser import validate_identifier
from app.sql.validator import to_jsonable

__all__ = ["Caller", "describe_table", "export_csv", "list_schemas", "list_tables", "table_rows"]

REDACTED = "***"


def _ro() -> Connection:
    return get_read_engine().connect()


def _own_fk(caller: Caller, ref: str | None) -> str | None:
    """Foreign keys are shown with logical names, and only if they stay in the workspace."""
    if not ref:
        return None
    ref_schema, rest = ref.split(".", 1)
    logical = to_logical(caller, ref_schema)
    return f"{logical}.{rest}" if logical else None


def _resolve_schema(conn: Connection, caller: Caller, schema: str) -> str:
    physical = to_physical(caller, schema)
    if not schema_service.schema_exists(conn, physical):
        raise schema_not_found(schema)
    return physical


def _load_table(conn: Connection, caller: Caller, schema: str, tbl: str) -> CatalogTable:
    physical = to_physical(caller, schema)
    try:
        validate_identifier(tbl, "table")
    except AppError:
        raise table_not_found(schema, tbl) from None
    info = schema_service.get_table(conn, physical, tbl)
    if info is None:
        raise table_not_found(schema, tbl)
    return info


# --------------------------------------------------------------------------- #
def list_schemas(caller: Caller) -> dict[str, Any]:
    with _ro() as conn:
        items: list[dict[str, Any]] = []
        for physical in schema_service.owned_schemas(conn, caller.prefix):
            logical = to_logical(caller, physical)
            if logical:
                items.append({"schema": logical,
                              "table_count": len(schema_service.list_tables(conn, physical))})
    names = ", ".join(i["schema"] for i in items)
    message = (f"You have {len(items)} schema(s): {names}." if items else
               f"Your workspace is empty. Create a table or import a file to start "
               f"(the default schema is {settings.default_schema}).")
    return {
        "type": "schema_list",
        "data": {"columns": ["schema", "table_count"], "items": items},
        "message": message,
    }


def list_tables(caller: Caller, schema: str) -> dict[str, Any]:
    with _ro() as conn:
        physical = _resolve_schema(conn, caller, schema)
        tables = schema_service.list_tables(conn, physical)
    items = [{"table": t.name, "kind": t.kind, "estimated_rows": t.estimated_rows} for t in tables]
    names = ", ".join(t.name for t in tables) or "no tables"
    return {
        "type": "table_list",
        "data": {"schema": schema, "columns": ["table", "kind", "estimated_rows"], "items": items},
        "message": f"Schema {schema} has {len(items)} table(s): {names}.",
    }


def describe_table(caller: Caller, schema: str, tbl: str) -> dict[str, Any]:
    with _ro() as conn:
        info = _load_table(conn, caller, schema, tbl)
        row_count = info.estimated_rows
        exact = False
        if row_count is None:  # never analyzed: count, but cheaply capped
            capped = 100_000
            sub: Any = select(literal_column("1")).select_from(
                table(tbl, schema=info.schema)).limit(capped + 1)
            row_count = int(conn.execute(select(func.count()).select_from(sub.subquery())).scalar_one())
            exact = row_count <= capped
        sample: list[dict[str, Any]] = []
        if settings.describe_sample_rows:
            sample = _rows(conn, info, limit=settings.describe_sample_rows, offset=0)

    columns = [
        {
            "name": c.name,
            "type": c.display_type,
            "nullable": c.nullable,
            "default": c.default,
            "primary_key": c.primary_key,
            "references": _own_fk(caller, c.references),
            "sensitive": bool(SENSITIVE_COLUMN_RE.search(c.name)),
        }
        for c in info.columns
    ]
    pk = [c.name for c in info.columns if c.primary_key]
    summary = (
        f"{schema}.{tbl} has {len(columns)} column(s): "
        + ", ".join(f"{c['name']} ({c['type']})" for c in columns)
        + (f". Primary key: {', '.join(pk)}" if pk else "")
        + f". About {row_count} row(s)."
    )
    return {
        "type": "table_description",
        "data": {
            "schema": schema,
            "table": tbl,
            "kind": info.kind,
            "columns": columns,
            "primary_key": pk,
            "row_count": row_count,
            "row_count_is_estimate": not exact and info.estimated_rows is not None,
            "sample_rows": sample,
        },
        "message": summary,
    }


# --------------------------------------------------------------------------- #
def _rows(conn: Connection, info: CatalogTable, *, limit: int, offset: int,
          sort: str | None = None, direction: Literal["asc", "desc"] = "asc") -> list[dict]:
    names = [c.name for c in info.columns]
    tbl = table(info.name, *[column(n) for n in names], schema=info.schema)
    stmt = select(*[tbl.c[n] for n in names]).limit(limit).offset(offset)
    if sort:
        stmt = stmt.order_by(desc(tbl.c[sort]) if direction == "desc" else asc(tbl.c[sort]))
    elif any(c.primary_key for c in info.columns):
        stmt = stmt.order_by(*[tbl.c[c.name] for c in info.columns if c.primary_key])
    out = []
    for row in conn.execute(stmt).mappings():
        out.append({k: REDACTED if SENSITIVE_COLUMN_RE.search(k) else to_jsonable(v)
                    for k, v in row.items()})
    return out


def table_rows(caller: Caller, schema: str, tbl: str, *, page: int = 1, page_size: int = 50,
               sort: str | None = None, direction: str = "asc") -> dict[str, Any]:
    page = max(page, 1)
    page_size = max(1, min(page_size, settings.page_size_max))
    if direction not in ("asc", "desc"):
        raise AppError(ErrorCode.VALIDATION_ERROR, "dir must be 'asc' or 'desc'.")
    with _ro() as conn:
        info = _load_table(conn, caller, schema, tbl)
        names = [c.name for c in info.columns]
        if sort is not None and sort not in names:  # whitelist against real columns
            raise AppError(ErrorCode.COLUMN_NOT_FOUND, f"Cannot sort by {sort[:63]!r}: no such column.",
                           details={"columns": names})
        rows = _rows(conn, info, limit=page_size, offset=(page - 1) * page_size,
                     sort=sort, direction=direction)  # type: ignore[arg-type]
        total = int(conn.execute(
            select(func.count()).select_from(table(tbl, schema=info.schema))
        ).scalar_one())
    return {
        "type": "query_result",
        "data": {"schema": schema, "table": tbl, "columns": names, "rows": rows,
                 "page": page, "page_size": page_size, "total_rows": total,
                 "has_more": page * page_size < total},
        "message": f"Rows {min((page - 1) * page_size + 1, total)}-"
                   f"{min(page * page_size, total)} of {total}.",
    }


def export_csv(caller: Caller, schema: str, tbl: str) -> Iterator[str]:
    """Streams CSV text in chunks; capped at CSV_EXPORT_MAX_ROWS rows.

    The table is resolved before the first chunk, so errors become normal HTTP errors.
    """
    with _ro() as conn:
        info = _load_table(conn, caller, schema, tbl)
    names = [c.name for c in info.columns]
    cap = settings.csv_export_max_rows

    def generate() -> Iterator[str]:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(names)
        sent = 0
        tbl_ = table(info.name, *[column(n) for n in names], schema=info.schema)
        stmt = select(*[tbl_.c[n] for n in names]).limit(cap)
        with get_read_engine().connect() as conn2:
            result = conn2.execution_options(stream_results=True, max_row_buffer=1000).execute(stmt)
            for row in result:
                writer.writerow([
                    REDACTED if SENSITIVE_COLUMN_RE.search(n) else _csv_cell(v)
                    for n, v in zip(names, row, strict=True)
                ])
                sent += 1
                if buf.tell() > 64_000:
                    yield buf.getvalue()
                    buf.seek(0)
                    buf.truncate()
        if sent >= cap:
            writer.writerow([f"# truncated at {cap} rows"])
        yield buf.getvalue()

    return generate()


def _csv_cell(value: Any) -> Any:
    value = to_jsonable(value)
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value  # stop spreadsheet formula injection
    return value


def quick_health() -> bool:
    try:
        with _ro() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
