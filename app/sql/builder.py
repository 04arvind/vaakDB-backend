"""SQLAlchemy Core builders for validated plans.

No SQL is ever assembled from strings: identifiers go through SQLAlchemy's
quoting (and were already checked against ^[a-z_][a-z0-9_]{0,62}$), values are
bound parameters. `render()` is for human-readable previews only and is never
executed.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    ColumnElement,
    Date,
    DateTime,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    and_,
    column,
    table,
    true,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable, DropTable
from sqlalchemy.sql import ClauseElement
from sqlalchemy.sql.expression import TableClause
from sqlalchemy.types import TypeEngine

from app.schemas.plans import Condition
from app.sql.parser import PgType
from app.sql.validator import CastError, ColumnSpec, cast_value

_DIALECT = postgresql.dialect()  # type: ignore[no-untyped-call]


def sa_type(t: PgType) -> TypeEngine[Any]:
    if t.name == "text":
        return Text()
    if t.name == "varchar":
        return String(t.length)
    if t.name == "integer":
        return Integer()
    if t.name == "bigint":
        return BigInteger()
    if t.name == "numeric":
        return Numeric(t.precision, t.scale) if t.precision else Numeric()
    if t.name == "boolean":
        return Boolean()
    if t.name == "date":
        return Date()
    if t.name == "timestamp":
        return DateTime(timezone=False)
    if t.name == "uuid":
        return postgresql.UUID(as_uuid=True)
    raise ValueError(f"unsupported type {t}")


def ddl_table(schema: str, name: str, columns: list[ColumnSpec], primary_key: list[str]) -> Table:
    md = MetaData(schema=schema)
    return Table(
        name,
        md,
        *[
            Column(c.name, sa_type(c.type), nullable=c.nullable,
                   primary_key=c.name in primary_key, autoincrement=False)
            for c in columns
        ],
    )


def dml_table(schema: str, name: str, columns: list[ColumnSpec]) -> TableClause:
    """Lightweight table for INSERT/UPDATE/DELETE with typed columns (bind processing)."""
    return table(name, *[column(c.name, sa_type(c.type)) for c in columns], schema=schema)


def create_table_stmt(tbl: Table) -> CreateTable:
    return CreateTable(tbl)


def drop_table_stmt(tbl: Table) -> DropTable:
    return DropTable(tbl)


def where_clause(tbl: TableClause, conditions: list[Condition],
                 types: dict[str, PgType]) -> ColumnElement[bool]:
    """AND of all conditions; values cast to the column type first (CastError on mismatch)."""
    parts: list[ColumnElement[bool]] = []
    for cond in conditions:
        col = tbl.c[cond.column]
        t = types[cond.column]
        if cond.op == "is_null":
            parts.append(col.is_(None))
            continue
        if cond.op == "is_not_null":
            parts.append(col.is_not(None))
            continue
        if cond.op in ("in", "not_in"):
            values = [cast_value(v, t) for v in cond.value]  # type: ignore[union-attr]
            if any(v is None for v in values):
                raise CastError("IN lists cannot contain NULL; use is_null")
            parts.append(col.in_(values) if cond.op == "in" else col.not_in(values))
            continue
        value = cast_value(cond.value, t)
        if value is None:
            raise CastError(f"value for {cond.column} is empty; use is_null")
        if cond.op == "like":
            if t.name not in ("text", "varchar"):
                raise CastError(f"'like' only works on text columns, not {t}")
            parts.append(col.like(value))
        elif cond.op == "=":
            parts.append(col == value)
        elif cond.op == "!=":
            parts.append(col != value)
        elif cond.op == "<":
            parts.append(col < value)
        elif cond.op == "<=":
            parts.append(col <= value)
        elif cond.op == ">":
            parts.append(col > value)
        elif cond.op == ">=":
            parts.append(col >= value)
    return and_(*parts) if parts else true()


def render(stmt: ClauseElement, *, literal: bool = False) -> str:
    """Human-readable SQL for previews (never executed)."""
    try:
        compiled = stmt.compile(dialect=_DIALECT, compile_kwargs={"literal_binds": literal})
    except Exception:
        compiled = stmt.compile(dialect=_DIALECT)
    return str(compiled).strip()
