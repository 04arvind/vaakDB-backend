"""Plan validator: checks a structured plan against the caller's own workspace,
casts every value, and produces the preview.

Plans use the user's logical schema names ("college"); the validator maps them
to the caller's private physical schema ("u7_college") and every lookup and
statement uses that. Previews and error messages show logical names again.

Runs at plan time (to build the preview) and again at confirm time (the
database may have changed), so execution only ever uses a freshly validated plan.

Errors raise PlanValidationError (an AppError). `repairable=True` means the LLM
may be able to fix its plan (bad type, unknown column ...); permission and
schema errors are never sent back for "repair".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Connection, func, insert, select, update
from sqlalchemy import delete as sa_delete

from app.core.config import settings
from app.core.exceptions import AppError, ErrorCode
from app.schemas.plans import (
    AnyPlan,
    CreateTablePlan,
    DeleteRowsPlan,
    InsertRowsPlan,
    UpdateRowsPlan,
)
from app.security.authorization import Caller, logicalize_text, table_not_found, to_physical
from app.services import schema_service
from app.services.schema_service import CatalogTable
from app.sql import builder
from app.sql.parser import PgType, parse_pg_type, validate_identifier
from app.sql.policy import check_plan_safety, destructive_reasons
from app.sql.validator import CastError, CastReport, ColumnSpec, cast_rows, cast_value, to_jsonable


class PlanValidationError(AppError):
    def __init__(self, code: ErrorCode, message: str, *, repairable: bool = False,
                 details: dict[str, Any] | None = None):
        super().__init__(code, message, details={**(details or {}), "repairable": repairable})
        self.repairable = repairable


@dataclass
class ValidatedPlan:
    plan: AnyPlan
    table_exists: bool
    columns: list[ColumnSpec]  # the columns being written, with their types
    primary_key: list[str] = field(default_factory=list)
    cast: CastReport = field(default_factory=CastReport)
    affected_rows: int | None = None
    warnings: list[str] = field(default_factory=list)
    statements: list[str] = field(default_factory=list)  # human-readable SQL
    column_mapping: list[dict[str, Any]] = field(default_factory=list)
    existing: CatalogTable | None = None
    set_values: dict[str, Any] = field(default_factory=dict)  # update: typed SET values
    column_types: dict[str, PgType] = field(default_factory=dict)  # update/delete: SET + WHERE
    # the plan as the user wrote it (logical schema); `plan` holds the physical one
    source_plan: AnyPlan | None = None
    logical_schema: str = ""
    create_schema: bool = False  # first table in a new schema of this workspace

    @property
    def rows_to_write(self) -> int:
        return len(self.cast.rows)

    def preview(self) -> dict[str, Any]:
        plan = self.plan
        total = len(plan.rows) if isinstance(plan, (CreateTablePlan, InsertRowsPlan)) else None
        names = [c.name for c in self.columns]
        return {
            "operation_type": plan.kind,
            "target": {"schema": self.logical_schema or plan.schema_name, "table": plan.table},
            "mode": getattr(plan, "if_exists", None),
            "table_exists": self.table_exists,
            "ddl": self.statements,
            "columns": [{"name": c.name, "type": str(c.type), "nullable": c.nullable,
                         "primary_key": c.name in self.primary_key} for c in self.columns],
            "column_mapping": self.column_mapping,
            "sample_rows": [dict(zip(names, [to_jsonable(v) for v in r], strict=True))
                            for r in self.cast.rows[: settings.preview_rows]],
            "total_rows": total,
            "rows_to_write": self.rows_to_write if total is not None else None,
            "failed_rows": self.cast.failures,
            "failed_row_count": self.cast.failed_row_count,
            "affected_rows": self.affected_rows,
            "destructive": destructive_reasons(plan, self.logical_schema or None),
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- #
def _fail(code: ErrorCode, message: str, *, repairable: bool = False, **details: Any) -> PlanValidationError:
    return PlanValidationError(code, message, repairable=repairable, details=details)


def _check_identifiers(plan: AnyPlan) -> None:
    """Defence in depth: the Pydantic model already normalized these."""
    validate_identifier(plan.schema_name, "schema")
    validate_identifier(plan.table, "table")
    if isinstance(plan, CreateTablePlan):
        for c in plan.columns:
            validate_identifier(c.name, "column")
    elif isinstance(plan, InsertRowsPlan):
        for name in plan.columns:
            validate_identifier(name, "column")
    elif isinstance(plan, UpdateRowsPlan):
        for a in plan.set:
            validate_identifier(a.column, "column")
    if isinstance(plan, (UpdateRowsPlan, DeleteRowsPlan)):
        for cond in plan.where:
            validate_identifier(cond.column, "column")


def _existing_specs(info: CatalogTable) -> dict[str, tuple[ColumnSpec, bool]]:
    """column -> (spec, writable). Columns of types we don't write (json, arrays ...)
    are listed as not writable."""
    out: dict[str, tuple[ColumnSpec, bool]] = {}
    for c in info.columns:
        t = c.pg_type
        out[c.name] = (ColumnSpec(c.name, t or PgType("text"), c.nullable), t is not None)
    return out


def _required_columns(info: CatalogTable) -> set[str]:
    return {c.name for c in info.columns if not c.nullable and c.default is None}


def _row_cap(rows: list[list[Any]]) -> None:
    if len(rows) > settings.max_insert_rows:
        raise _fail(ErrorCode.ROW_LIMIT_EXCEEDED,
                    f"{len(rows)} rows is more than the limit of {settings.max_insert_rows} "
                    "rows per operation. Split the data into smaller files.",
                    rows=len(rows), limit=settings.max_insert_rows)


def _dedupe_primary_key(report: CastReport, columns: list[ColumnSpec], pk: list[str]) -> None:
    if not pk:
        return
    idx = [i for i, c in enumerate(columns) if c.name in pk]
    seen: set[tuple[Any, ...]] = set()
    keep_rows, keep_numbers = [], []
    for row, number in zip(report.rows, report.row_numbers, strict=True):
        key = tuple(row[i] for i in idx)
        if key in seen:
            report.failed_row_count += 1
            if len(report.failures) < CastReport.MAX_FAILURES_REPORTED:
                report.failures.append({"row": number, "column": ",".join(pk),
                                        "value": str(key[0] if len(key) == 1 else key)[:100],
                                        "error": "duplicate primary key value"})
            continue
        seen.add(key)
        keep_rows.append(row)
        keep_numbers.append(number)
    report.rows, report.row_numbers = keep_rows, keep_numbers


def _cast_or_fail(columns: list[ColumnSpec], rows: list[list[Any]]) -> CastReport:
    report = cast_rows(columns, rows)
    if rows and not report.rows:
        raise _fail(ErrorCode.TYPE_CAST_FAILED,
                    "None of the rows match the column types. Check the column types.",
                    repairable=True, failed_rows=report.failures[:20])
    return report


def _cast_warning(v: ValidatedPlan) -> None:
    if v.cast.failed_row_count:
        v.warnings.append(
            f"{v.cast.failed_row_count} row(s) do not match the column types and will be "
            "skipped (see failed_rows)."
        )


# --------------------------------------------------------------------------- #
def validate_plan(plan: AnyPlan, *, caller: Caller, conn: Connection) -> ValidatedPlan:
    """`plan` uses logical schema names; `conn` is a read-only target_db connection."""
    _check_identifiers(plan)
    check_plan_safety(plan)
    logical = plan.schema_name
    physical = to_physical(caller, logical)  # the only schema this caller can reach
    try:
        v = _validate_physical(plan.model_copy(update={"schema_name": physical}), caller, conn)
    except AppError as exc:
        exc.message = logicalize_text(caller, exc.message)
        raise
    v.source_plan = plan
    v.logical_schema = logical
    v.statements = [logicalize_text(caller, s) for s in v.statements]
    v.warnings = [logicalize_text(caller, w) for w in v.warnings]
    return v


def _validate_physical(plan: AnyPlan, caller: Caller, conn: Connection) -> ValidatedPlan:
    schema_exists = schema_service.schema_exists(conn, plan.schema_name)
    info = schema_service.get_table(conn, plan.schema_name, plan.table) if schema_exists else None
    if info is not None and info.kind != "table":
        raise _fail(ErrorCode.PLAN_INVALID, f"{plan.qualified} is a {info.kind}, not a table.")

    if isinstance(plan, CreateTablePlan):
        if not schema_exists:
            owned = schema_service.owned_schemas(conn, caller.prefix)
            if len(owned) >= settings.max_schemas_per_user:
                raise _fail(ErrorCode.PLAN_INVALID,
                            f"You already have {len(owned)} schemas (the limit is "
                            f"{settings.max_schemas_per_user}). Use an existing schema.")
        v = _validate_create(plan, info, conn)
        if not schema_exists:
            v.create_schema = True
            v.statements.insert(0, f"CREATE SCHEMA {plan.schema_name}")
            v.warnings.append(f"Schema {plan.schema_name} is new and will be created.")
        return v
    if info is None:
        raise table_not_found(plan.schema_name, plan.table)
    if isinstance(plan, InsertRowsPlan):
        return _validate_insert(plan, info)
    if isinstance(plan, UpdateRowsPlan):
        return _validate_update(plan, info, conn)
    if isinstance(plan, DeleteRowsPlan):
        return _validate_delete(plan, info, conn)
    raise _fail(ErrorCode.PLAN_INVALID, "Unknown plan type.")


def _mapping(plan: CreateTablePlan) -> list[dict[str, Any]]:
    return [{"source": c.source, "column": c.name, "type": c.pg_type} for c in plan.columns]


def _validate_create(plan: CreateTablePlan, info: CatalogTable | None,
                     conn: Connection) -> ValidatedPlan:
    _row_cap(plan.rows)
    warnings: list[str] = []
    if info is not None:
        if plan.if_exists == "fail":
            raise _fail(ErrorCode.TABLE_EXISTS,
                        f"Table {plan.qualified} already exists. Say whether to append the rows "
                        "to it or replace it (replace is admin-only).",
                        options=["append", "replace"])
        if plan.if_exists == "append":
            existing = _existing_specs(info)
            missing = [c.name for c in plan.columns if c.name not in existing]
            if missing:
                raise _fail(ErrorCode.COLUMN_NOT_FOUND,
                            f"Cannot append: {plan.qualified} has no column(s) {', '.join(missing)}.",
                            repairable=True, missing=missing,
                            existing=[c.name for c in info.columns])
            not_writable = [c.name for c in plan.columns if not existing[c.name][1]]
            if not_writable:
                raise _fail(ErrorCode.PLAN_INVALID,
                            f"Column(s) {', '.join(not_writable)} have a type that cannot be written.")
            required = _required_columns(info) - set(plan.column_names)
            if required:
                raise _fail(ErrorCode.PLAN_INVALID,
                            f"Cannot append: required column(s) {', '.join(sorted(required))} "
                            "are missing from the data.", repairable=True)
            columns = [existing[c.name][0] for c in plan.columns]
            for planned, actual in zip(plan.columns, columns, strict=True):
                if parse_pg_type(planned.pg_type) != actual.type:
                    warnings.append(f"{actual.name}: values are cast to the existing type "
                                    f"{actual.type} (plan said {planned.pg_type}).")
            v = ValidatedPlan(plan=plan, table_exists=True, columns=columns,
                              primary_key=[c.name for c in info.columns if c.primary_key],
                              existing=info, warnings=warnings, column_mapping=_mapping(plan))
            v.cast = _cast_or_fail(columns, plan.rows)
            _dedupe_primary_key(v.cast, columns, v.primary_key)
            tbl = builder.dml_table(plan.schema_name, plan.table, columns)
            v.statements = [builder.render(insert(tbl)) + f"  -- {v.rows_to_write} row(s)"]
            _cast_warning(v)
            return v
        warnings.append(f"Table {plan.qualified} exists and will be DROPPED and recreated.")
    elif plan.if_exists in ("append", "replace"):
        warnings.append(f"Table {plan.qualified} does not exist yet; it will be created.")

    columns = [ColumnSpec(c.name, parse_pg_type(c.pg_type), c.nullable) for c in plan.columns]
    pk = [c.name for c in plan.columns if c.primary_key]
    v = ValidatedPlan(plan=plan, table_exists=info is not None, columns=columns, primary_key=pk,
                      existing=info, warnings=warnings, column_mapping=_mapping(plan))
    v.cast = _cast_or_fail(columns, plan.rows)
    _dedupe_primary_key(v.cast, columns, pk)
    ddl = builder.ddl_table(plan.schema_name, plan.table, columns, pk)
    statements = []
    if info is not None:  # replace
        statements.append(builder.render(builder.drop_table_stmt(ddl)))
    statements.append(builder.render(builder.create_table_stmt(ddl)))
    if plan.rows:
        statements.append(builder.render(insert(ddl)) + f"  -- {v.rows_to_write} row(s)")
    v.statements = statements
    if not pk:
        v.warnings.append("The table has no primary key.")
    _cast_warning(v)
    return v


def _columns_for(info: CatalogTable, names: list[str]) -> list[ColumnSpec]:
    existing = _existing_specs(info)
    missing = [n for n in names if n not in existing]
    if missing:
        raise _fail(ErrorCode.COLUMN_NOT_FOUND,
                    f"{info.schema}.{info.name} has no column(s) {', '.join(missing)}.",
                    repairable=True, missing=missing, existing=list(existing))
    bad = [n for n in names if not existing[n][1]]
    if bad:
        raise _fail(ErrorCode.PLAN_INVALID,
                    f"Column(s) {', '.join(bad)} have a type that cannot be written through chat.")
    return [existing[n][0] for n in names]


def _validate_insert(plan: InsertRowsPlan, info: CatalogTable) -> ValidatedPlan:
    _row_cap(plan.rows)
    columns = _columns_for(info, plan.columns)
    required = _required_columns(info) - set(plan.columns)
    if required:
        raise _fail(ErrorCode.PLAN_INVALID,
                    f"Required column(s) {', '.join(sorted(required))} have no value.",
                    repairable=True)
    v = ValidatedPlan(plan=plan, table_exists=True, columns=columns, existing=info,
                      primary_key=[c.name for c in info.columns if c.primary_key])
    v.cast = _cast_or_fail(columns, plan.rows)
    _dedupe_primary_key(v.cast, columns, [p for p in v.primary_key if p in plan.columns])
    tbl = builder.dml_table(plan.schema_name, plan.table, columns)
    v.statements = [builder.render(insert(tbl)) + f"  -- {v.rows_to_write} row(s)"]
    _cast_warning(v)
    return v


def _where(plan: UpdateRowsPlan | DeleteRowsPlan, info: CatalogTable,
           extra: list[str]) -> tuple[Any, Any, dict[str, PgType]]:
    names = list(dict.fromkeys([*extra, *(c.column for c in plan.where)]))
    specs = _columns_for(info, names) if names else []
    types = {s.name: s.type for s in specs}
    tbl = builder.dml_table(plan.schema_name, plan.table, specs)
    try:
        clause = builder.where_clause(tbl, plan.where, types)
    except CastError as exc:
        raise _fail(ErrorCode.TYPE_CAST_FAILED, f"WHERE condition: {exc}", repairable=True) from None
    return tbl, clause, types


def _count(conn: Connection, tbl: Any, clause: Any) -> int:
    return int(conn.execute(select(func.count()).select_from(tbl).where(clause)).scalar_one())


def _validate_update(plan: UpdateRowsPlan, info: CatalogTable, conn: Connection) -> ValidatedPlan:
    set_names = [a.column for a in plan.set]
    tbl, clause, types = _where(plan, info, set_names)
    values: dict[str, Any] = {}
    nullable = {c.name: c.nullable for c in info.columns}
    for a in plan.set:
        try:
            value = cast_value(a.value, types[a.column])
        except CastError as exc:
            raise _fail(ErrorCode.TYPE_CAST_FAILED, f"SET {a.column}: {exc}", repairable=True) from None
        if value is None and not nullable.get(a.column, True):
            raise _fail(ErrorCode.TYPE_CAST_FAILED, f"SET {a.column}: column cannot be NULL.",
                        repairable=True)
        values[a.column] = value
    affected = _count(conn, tbl, clause)
    v = ValidatedPlan(plan=plan, table_exists=True, existing=info,
                      columns=[ColumnSpec(n, types[n]) for n in set_names],
                      affected_rows=affected, set_values=values, column_types=types)
    v.statements = [builder.render(update(tbl).where(clause).values(**values), literal=True)]
    if affected == 0:
        v.warnings.append("No rows match the condition; nothing would change.")
    return v


def _validate_delete(plan: DeleteRowsPlan, info: CatalogTable, conn: Connection) -> ValidatedPlan:
    tbl, clause, types = _where(plan, info, [])
    affected = _count(conn, tbl, clause)
    v = ValidatedPlan(plan=plan, table_exists=True, existing=info, columns=[],
                      affected_rows=affected, column_types=types)
    v.statements = [builder.render(sa_delete(tbl).where(clause), literal=True)]
    if affected == 0:
        v.warnings.append("No rows match the condition; nothing would be deleted.")
    return v
