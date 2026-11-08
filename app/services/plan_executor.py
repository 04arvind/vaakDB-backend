"""Plan executor: turns a validated plan into SQLAlchemy Core statements and
runs them on target_db as the restricted write role (vaak_rw).

- one transaction per operation: CREATE TABLE + all INSERT batches commit
  together or not at all (Postgres DDL is transactional), so a crash or a kill
  mid-insert leaves no table and no rows behind
- statement_timeout / lock_timeout set with set_config(..., is_local => true)
- row cap re-checked (MAX_INSERT_ROWS), batched executemany (INSERT_BATCH_SIZE, default 500)
- driver errors are translated to friendly messages; SQL text never leaves the server
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy import Connection, insert, select, text, update
from sqlalchemy import delete as sa_delete
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
from sqlalchemy.schema import CreateSchema

from app.core.config import settings
from app.core.exceptions import AppError, ErrorCode
from app.db.target_db import get_write_engine
from app.schemas.plans import CreateTablePlan, DeleteRowsPlan, InsertRowsPlan, UpdateRowsPlan
from app.services.plan_validator import ValidatedPlan
from app.sql import builder
from app.sql.validator import ColumnSpec

log = logging.getLogger(__name__)

# SQLSTATE -> (code, message)
_SQLSTATE: dict[str, tuple[ErrorCode, str]] = {
    "23505": (ErrorCode.EXECUTION_FAILED, "Some rows duplicate an existing primary key or unique value."),
    "23502": (ErrorCode.EXECUTION_FAILED, "A required column was left empty."),
    "23503": (ErrorCode.EXECUTION_FAILED, "A value does not match the referenced table (foreign key)."),
    "23514": (ErrorCode.EXECUTION_FAILED, "A value breaks a CHECK constraint on the table."),
    "22001": (ErrorCode.TYPE_CAST_FAILED, "A value is too long for its column."),
    "22003": (ErrorCode.TYPE_CAST_FAILED, "A number is out of range for its column."),
    "22P02": (ErrorCode.TYPE_CAST_FAILED, "A value has the wrong format for its column."),
    "42P07": (ErrorCode.TABLE_EXISTS, "The table was created by someone else in the meantime."),
    "42P01": (ErrorCode.TABLE_NOT_FOUND, "The table no longer exists."),
    "42703": (ErrorCode.COLUMN_NOT_FOUND, "A column no longer exists."),
    "42501": (ErrorCode.PERMISSION_DENIED,
              "The database write role is not allowed to do this (check scripts/create_rw_role.sql)."),
    "57014": (ErrorCode.EXECUTION_FAILED, "The operation took too long and was cancelled."),
    "55P03": (ErrorCode.EXECUTION_FAILED, "The table is locked by another operation; try again."),
    "40P01": (ErrorCode.EXECUTION_FAILED, "A deadlock occurred; try again."),
}

AfterBatch = Callable[[int, int], None]  # (batch_number, rows_so_far)


def _translate(exc: Exception) -> AppError:
    sqlstate = getattr(getattr(exc, "orig", None), "sqlstate", None)
    code, message = _SQLSTATE.get(sqlstate or "", (ErrorCode.EXECUTION_FAILED, ""))
    rolled = " The operation was rolled back; no data was changed."
    return AppError(code, (message or "The database rejected the operation.") + rolled,
                    details={"sqlstate": sqlstate} if sqlstate else None)


def _set_limits(conn: Connection) -> None:
    conn.execute(select(text("set_config('statement_timeout', :v, true)")),
                 {"v": str(settings.statement_timeout_ms)})
    conn.execute(select(text("set_config('lock_timeout', :v, true)")),
                 {"v": str(settings.lock_timeout_ms)})


def _insert_batches(conn: Connection, tbl: Any, names: list[str], rows: list[list[Any]],
                    after_batch: AfterBatch | None) -> int:
    size = settings.insert_batch_size
    written = 0
    for number, start in enumerate(range(0, len(rows), size), start=1):
        chunk = [dict(zip(names, row, strict=True)) for row in rows[start:start + size]]
        conn.execute(insert(tbl), chunk)  # executemany with bound parameters
        written += len(chunk)
        if after_batch is not None:
            after_batch(number, written)
    return written


def execute_plan(v: ValidatedPlan, *, after_batch: AfterBatch | None = None) -> dict[str, Any]:
    """Run a validated plan in ONE transaction. Returns a JSON-safe result."""
    plan = v.plan
    if v.rows_to_write > settings.max_insert_rows:
        raise AppError(ErrorCode.ROW_LIMIT_EXCEEDED,
                       f"At most {settings.max_insert_rows} rows can be written at once.")
    result: dict[str, Any] = {"operation_type": plan.kind,
                              "target": {"schema": v.logical_schema or plan.schema_name,
                                         "table": plan.table}}
    names = [c.name for c in v.columns]
    try:
        with get_write_engine().begin() as conn:
            _set_limits(conn)
            if v.create_schema:  # first table in a new workspace schema (same transaction)
                conn.execute(CreateSchema(plan.schema_name, if_not_exists=True))
                result["schema_created"] = True
            if isinstance(plan, CreateTablePlan):
                if v.table_exists and plan.if_exists == "append":
                    tbl = builder.dml_table(plan.schema_name, plan.table, v.columns)
                    result["table_created"] = False
                else:
                    tbl = builder.ddl_table(plan.schema_name, plan.table, v.columns, v.primary_key)
                    if v.table_exists:  # replace (admin only, checked by the validator)
                        conn.execute(builder.drop_table_stmt(tbl))
                        result["table_replaced"] = True
                    conn.execute(builder.create_table_stmt(tbl))
                    result["table_created"] = True
                result["rows_written"] = _insert_batches(conn, tbl, names, v.cast.rows, after_batch)
            elif isinstance(plan, InsertRowsPlan):
                tbl = builder.dml_table(plan.schema_name, plan.table, v.columns)
                result["rows_written"] = _insert_batches(conn, tbl, names, v.cast.rows, after_batch)
            elif isinstance(plan, (UpdateRowsPlan, DeleteRowsPlan)):
                specs = [ColumnSpec(n, t) for n, t in v.column_types.items()]
                tbl = builder.dml_table(plan.schema_name, plan.table, specs)
                clause = builder.where_clause(tbl, plan.where, v.column_types)
                if isinstance(plan, UpdateRowsPlan):
                    res = conn.execute(update(tbl).where(clause).values(**v.set_values))
                    result["rows_updated"] = res.rowcount
                else:
                    res = conn.execute(sa_delete(tbl).where(clause))
                    result["rows_deleted"] = res.rowcount
                result["preview_affected_rows"] = v.affected_rows
            else:  # pragma: no cover
                raise AppError(ErrorCode.PLAN_INVALID, "Unknown plan type.")
    except AppError:
        raise
    except DBAPIError as exc:
        log.warning("Execution failed on %s.%s: sqlstate=%s", plan.schema_name, plan.table,
                    getattr(exc.orig, "sqlstate", None))
        raise _translate(exc) from None
    except SQLAlchemyError as exc:
        log.warning("Execution failed on %s.%s: %s", plan.schema_name, plan.table,
                    type(exc).__name__)
        raise _translate(exc) from None

    result["rows_skipped"] = v.cast.failed_row_count
    return result
