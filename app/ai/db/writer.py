"""Controlled writes. The LLM never writes SQL for writes.

Flow: extractor_agent -> validated Pydantic object -> insert_records(target, ...)

Only targets you register here can be written. Each target maps a logical name
(what the API/LLM sees, e.g. "expenses") to ONE table, ONE Pydantic model and an
explicit column list. SQL is built with SQLAlchemy Core (parameterized), and the
write engine should use a Postgres role granted INSERT/UPDATE on those tables only.

Register your tables at startup (for example in app/ai/db/targets.py):

    from pydantic import BaseModel
    from app.ai.db.writer import register_target

    class Expense(BaseModel):
        vendor: str | None = None
        amount: float | None = None
        spent_on: date | None = None

    register_target(
        "expenses", table="expenses", model=Expense,
        owner_column="user_id",            # set automatically from the caller
        conflict_columns=("user_id", "vendor", "amount", "spent_on"),  # skip duplicates
        updatable_columns=("vendor",),     # leave empty for insert-only
    )
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence

from pydantic import BaseModel, TypeAdapter, ValidationError
from sqlalchemy import column, table, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.ai.db.engine import get_write_engine
from app.ai.exceptions import UnknownTargetError

log = logging.getLogger(__name__)

_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")


@dataclass(frozen=True)
class WriteTarget:
    name: str
    table: str
    model: type[BaseModel]
    pk: str = "id"
    columns: tuple[str, ...] | None = None  # default: every field of the model
    owner_column: str | None = None
    conflict_columns: tuple[str, ...] = ()
    updatable_columns: tuple[str, ...] = ()
    max_records: int = 100
    hints: str = ""  # extra guidance for the extractor, e.g. "amounts are in INR"

    @property
    def insert_columns(self) -> tuple[str, ...]:
        cols = self.columns or tuple(self.model.model_fields)
        if self.owner_column and self.owner_column not in cols:
            cols = (*cols, self.owner_column)
        return cols


_TARGETS: dict[str, WriteTarget] = {}


def register_target(
    name: str,
    *,
    table: str,
    model: type[BaseModel],
    pk: str = "id",
    columns: Sequence[str] | None = None,
    owner_column: str | None = None,
    conflict_columns: Sequence[str] = (),
    updatable_columns: Sequence[str] = (),
    max_records: int = 100,
    hints: str = "",
) -> None:
    names = [name, table, pk, *(columns or ()), *conflict_columns, *updatable_columns]
    if owner_column:
        names.append(owner_column)
    for ident in names:
        if not _IDENT.match(ident):
            raise ValueError(f"Invalid identifier in write target '{name}': {ident!r}")

    fields = set(model.model_fields)
    for label, cols in (("columns", columns or ()), ("updatable_columns", updatable_columns)):
        missing = set(cols) - fields
        if missing:
            raise ValueError(f"{label} not in model for '{name}': {sorted(missing)}")
    if owner_column and owner_column in updatable_columns:
        raise ValueError(f"owner_column cannot be updatable for '{name}'.")
    if not 1 <= max_records <= 10_000:
        raise ValueError("max_records must be between 1 and 10000.")

    _TARGETS[name] = WriteTarget(
        name=name, table=table, model=model, pk=pk,
        columns=tuple(columns) if columns else None,
        owner_column=owner_column,
        conflict_columns=tuple(conflict_columns),
        updatable_columns=tuple(updatable_columns),
        max_records=max_records,
        hints=hints,
    )


def get_target(name: str) -> WriteTarget:
    try:
        return _TARGETS[name]
    except KeyError:
        raise UnknownTargetError(name) from None


def unregister_target(name: str) -> None:
    """Remove a target (mainly for tests)."""
    _TARGETS.pop(name, None)


def list_targets() -> list[str]:
    return sorted(_TARGETS)


# --------------------------------------------------------------------------- #
async def insert_records(
    target_name: str,
    records: Sequence[BaseModel | dict[str, Any]],
    *,
    user_id: str | None = None,
) -> list[int | str]:
    """Insert validated records atomically. Returns the new primary keys.

    Rows skipped by conflict_columns (duplicates) are not returned.
    Raises UnknownTargetError, or ValueError for bad input.
    """
    target = get_target(target_name)
    if not records:
        return []
    if len(records) > target.max_records:
        raise ValueError(f"Too many records (max {target.max_records}).")
    if target.owner_column and not user_id:
        raise ValueError("A user is required to write to this target.")

    allowed = set(target.insert_columns)
    rows: list[dict[str, Any]] = []
    for rec in records:
        obj = _as_model(target, rec)
        data = {k: v for k, v in obj.model_dump(mode="python").items() if k in allowed}
        if target.owner_column:
            data[target.owner_column] = user_id
        rows.append(data)

    # Same keys in every row (executemany); skip columns that are empty everywhere
    # so database defaults still apply.
    used = [c for c in target.insert_columns if any(r.get(c) is not None for r in rows)]
    if not any(c != target.owner_column for c in used):
        raise ValueError("Nothing to insert: all fields are empty.")
    rows = [{c: r.get(c) for c in used} for r in rows]

    tbl = table(target.table, *[column(c) for c in used], column(target.pk))
    stmt = pg_insert(tbl)
    if target.conflict_columns:
        stmt = stmt.on_conflict_do_nothing(index_elements=list(target.conflict_columns))
    stmt = stmt.returning(tbl.c[target.pk])

    async with get_write_engine().begin() as conn:
        result = await conn.execute(stmt, rows)
        ids = list(result.scalars().all())

    log.info("Inserted %d/%d rows into %s", len(ids), len(rows), target.table)
    return ids


async def update_record(
    target_name: str,
    record_id: int | str,
    changes: dict[str, Any],
    *,
    user_id: str | None = None,
) -> bool:
    """Update whitelisted columns of one row. Returns True if a row was changed."""
    target = get_target(target_name)
    if not target.updatable_columns:
        raise ValueError(f"Target '{target_name}' is insert-only.")
    if target.owner_column and not user_id:
        raise ValueError("A user is required to update this target.")

    bad = set(changes) - set(target.updatable_columns)
    if bad:
        raise ValueError(f"Columns not updatable: {sorted(bad)}")
    if not changes:
        raise ValueError("No changes provided.")

    clean: dict[str, Any] = {}
    for key, value in changes.items():
        annotation = target.model.model_fields[key].annotation
        try:
            clean[key] = TypeAdapter(annotation).validate_python(value)
        except ValidationError as exc:
            raise ValueError(f"Invalid value for '{key}'.") from exc

    cols = {target.pk, *clean}
    if target.owner_column:
        cols.add(target.owner_column)
    tbl = table(target.table, *[column(c) for c in sorted(cols)])

    stmt = update(tbl).where(tbl.c[target.pk] == record_id).values(**clean)
    if target.owner_column:
        stmt = stmt.where(tbl.c[target.owner_column] == user_id)

    async with get_write_engine().begin() as conn:
        result = await conn.execute(stmt)
    return result.rowcount > 0


def _as_model(target: WriteTarget, rec: BaseModel | dict[str, Any]) -> BaseModel:
    if isinstance(rec, target.model):
        return rec
    if isinstance(rec, dict):
        try:
            return target.model.model_validate(rec)
        except ValidationError as exc:
            raise ValueError(f"Record failed validation for '{target.name}'.") from exc
    raise TypeError(f"Expected {target.model.__name__} or dict, got {type(rec).__name__}")