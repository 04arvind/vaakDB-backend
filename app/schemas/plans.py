"""Structured plans. The LLM never writes SQL for writes: it fills one of these
models, Pydantic validates it, plan_validator checks it against the database
and the user's permissions, and plan_executor builds the SQL itself.

Identifiers are normalized ('Roll No' -> 'roll_no') and must then match
^[a-z_][a-z0-9_]{0,62}$. Values are JSON scalars only; they are always sent
to Postgres as bound parameters.
"""
from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Annotated, Any, Literal, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.core.exceptions import AppError
from app.sql.parser import canonical_type, normalize_identifier

MAX_PLAN_COLUMNS = 200
MAX_IN_VALUES = 1000


def _ident(kind: Literal["schema", "table", "column"]) -> Any:
    def check(value: Any) -> str:
        try:
            return normalize_identifier(
                value, kind, on_reserved="suffix" if kind == "column" else "reject"
            )
        except AppError as exc:  # pydantic only wraps ValueError
            raise ValueError(exc.message) from None

    return BeforeValidator(check)


def _type(value: Any) -> str:
    try:
        return canonical_type(value)
    except AppError as exc:
        raise ValueError(exc.message) from None


SchemaName = Annotated[str, _ident("schema")]
TableName = Annotated[str, _ident("table")]
ColumnName = Annotated[str, _ident("column")]
PgTypeName = Annotated[str, AfterValidator(_type)]
Scalar = Union[str, int, float, bool, None]


class Intent(StrEnum):
    list_schemas = "list_schemas"
    list_tables = "list_tables"
    describe_table = "describe_table"
    select_query = "select_query"
    ingest_file = "ingest_file"
    insert_rows = "insert_rows"
    update_rows = "update_rows"
    delete_rows = "delete_rows"
    create_table = "create_table"
    clarify = "clarify"
    unsupported = "unsupported"


WRITE_INTENTS = frozenset({Intent.ingest_file, Intent.insert_rows, Intent.update_rows,
                           Intent.delete_rows, Intent.create_table})
METADATA_INTENTS = frozenset({Intent.list_schemas, Intent.list_tables, Intent.describe_table})


class IntentDecision(BaseModel):
    """Output of the intent router."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    intent: Intent
    schema_name: str | None = Field(default=None, alias="schema", max_length=100)
    table: str | None = Field(default=None, max_length=100)
    clarification_question: str | None = Field(default=None, max_length=300)
    reason: str | None = Field(default=None, max_length=300)


# --------------------------------------------------------------------------- #
# plans
# --------------------------------------------------------------------------- #
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, str_strip_whitespace=True)


class _Target(_Strict):
    schema_name: SchemaName = Field(alias="schema")
    table: TableName

    @property
    def qualified(self) -> str:
        return f"{self.schema_name}.{self.table}"


class ColumnDef(_Strict):
    name: ColumnName
    pg_type: PgTypeName
    nullable: bool = True
    primary_key: bool = False
    # original header in the source file (shown in the column mapping)
    source: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _pk_not_null(self) -> ColumnDef:
        if self.primary_key:
            self.nullable = False
        return self


def _unique(names: list[str], what: str) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"Duplicate {what} {name!r}.")
        seen.add(name)


class CreateTablePlan(_Target):
    kind: Literal["create_table"] = "create_table"
    columns: list[ColumnDef] = Field(min_length=1, max_length=MAX_PLAN_COLUMNS)
    rows: list[list[Scalar]] = Field(default_factory=list)
    # what to do when the table exists: fail (default), append rows, replace (admin only)
    if_exists: Literal["fail", "append", "replace"] = "fail"
    upload_id: uuid.UUID | None = None
    source_table_index: int | None = Field(default=None, ge=0)

    @field_validator("columns")
    @classmethod
    def _unique_columns(cls, cols: list[ColumnDef]) -> list[ColumnDef]:
        _unique([c.name for c in cols], "column")
        return cols

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


class InsertRowsPlan(_Target):
    kind: Literal["insert_rows"] = "insert_rows"
    columns: list[ColumnName] = Field(min_length=1, max_length=MAX_PLAN_COLUMNS)
    rows: list[list[Scalar]] = Field(min_length=1)

    @field_validator("columns")
    @classmethod
    def _unique_columns(cls, cols: list[str]) -> list[str]:
        _unique(cols, "column")
        return cols


class Condition(_Strict):
    column: ColumnName
    op: Literal["=", "!=", "<", "<=", ">", ">=", "in", "not_in", "like", "is_null",
                "is_not_null"]
    value: Scalar | list[Scalar] = None

    @model_validator(mode="after")
    def _shape(self) -> Condition:
        if self.op in ("in", "not_in"):
            if not isinstance(self.value, list) or not 1 <= len(self.value) <= MAX_IN_VALUES:
                raise ValueError(f"'{self.op}' needs a list of 1-{MAX_IN_VALUES} values.")
        elif self.op in ("is_null", "is_not_null"):
            self.value = None
        else:
            if isinstance(self.value, list):
                raise ValueError(f"'{self.op}' needs a single value.")
            if self.value is None:
                raise ValueError(f"'{self.op}' needs a value; use is_null for NULL checks.")
        return self


class Assignment(_Strict):
    column: ColumnName
    value: Scalar


class UpdateRowsPlan(_Target):
    kind: Literal["update_rows"] = "update_rows"
    set: list[Assignment] = Field(min_length=1, max_length=MAX_PLAN_COLUMNS)
    where: list[Condition] = Field(default_factory=list, max_length=50)
    # Must be explicitly true to update every row (admin only).
    all_rows: bool = False

    @field_validator("set")
    @classmethod
    def _unique_columns(cls, items: list[Assignment]) -> list[Assignment]:
        _unique([a.column for a in items], "column in SET")
        return items


class DeleteRowsPlan(_Target):
    kind: Literal["delete_rows"] = "delete_rows"
    where: list[Condition] = Field(default_factory=list, max_length=50)
    all_rows: bool = False


Plan = Annotated[
    Union[CreateTablePlan, InsertRowsPlan, UpdateRowsPlan, DeleteRowsPlan],
    Field(discriminator="kind"),
]
AnyPlan = Union[CreateTablePlan, InsertRowsPlan, UpdateRowsPlan, DeleteRowsPlan]


class PlanEnvelope(BaseModel):
    plan: Plan


def parse_plan(data: dict[str, Any]) -> AnyPlan:
    """Rebuild a stored plan (operations.plan_json)."""
    return PlanEnvelope.model_validate({"plan": data}).plan


def dump_plan(plan: AnyPlan) -> dict[str, Any]:
    return plan.model_dump(mode="json", by_alias=True)


# --------------------------------------------------------------------------- #
# LLM draft for file ingestion: the model maps columns, the rows come from the file
# --------------------------------------------------------------------------- #
class IngestColumnDraft(_Strict):
    source_index: int = Field(ge=0, description="0-based index of the column in the file header")
    name: ColumnName
    pg_type: PgTypeName
    nullable: bool = True
    primary_key: bool = False


class IngestDraft(_Strict):
    schema_name: SchemaName = Field(alias="schema")
    table: TableName
    table_index: int = Field(default=0, ge=0, description="which extracted table to store")
    if_exists: Literal["fail", "append", "replace"] = "fail"
    columns: list[IngestColumnDraft] = Field(min_length=1, max_length=MAX_PLAN_COLUMNS)

    @field_validator("columns")
    @classmethod
    def _unique_columns(cls, cols: list[IngestColumnDraft]) -> list[IngestColumnDraft]:
        _unique([c.name for c in cols], "column")
        _unique([str(c.source_index) for c in cols], "source_index")
        return cols
