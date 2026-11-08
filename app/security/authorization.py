"""Personal workspaces: user-level data isolation in target_db.

There are no roles. Every signed-in user has full access (read, export, create,
insert, update, delete, replace) to their OWN workspace and no access to
anyone else's.

A workspace is the set of Postgres schemas whose name starts with the user's
private prefix:  u<user_id>_<name>. Users only ever see and type the logical
name ("college"); the backend maps it to the physical schema ("u7_college")
at every boundary, so a user cannot even name another user's schema:
- every logical name is validated and then prefixed with the caller's id
- every physical name read from the catalog is filtered by that prefix
- natural-language SQL is checked against the caller's own tables and its
  table references are rewritten to the caller's physical schema

App metadata in meta_db (uploads, operations, conversations, query history,
projects) is scoped by user_id in every query.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import SYSTEM_SCHEMAS, settings
from app.core.exceptions import AppError, ErrorCode
from app.sql.parser import validate_identifier

MAX_PHYSICAL_LEN = 63


@dataclass(frozen=True)
class Caller:
    """The signed-in user a request runs for."""

    user_id: int

    @property
    def prefix(self) -> str:
        return schema_prefix(self.user_id)


def schema_prefix(user_id: int) -> str:
    # The "_" after the digits keeps prefixes distinct: "u1_" never matches "u12_x".
    return f"u{int(user_id)}_"


def max_schema_name_length(user_id: int) -> int:
    return MAX_PHYSICAL_LEN - len(schema_prefix(user_id))


def to_physical(caller: Caller, logical: str) -> str:
    """'college' -> 'u7_college' for user 7. Raises INVALID_IDENTIFIER / FORBIDDEN_SCHEMA."""
    if not isinstance(logical, str) or not logical:
        raise AppError(ErrorCode.INVALID_IDENTIFIER, "A schema name is required.")
    if logical in SYSTEM_SCHEMAS or logical.startswith("pg_"):
        raise AppError(ErrorCode.FORBIDDEN_SCHEMA, "System schemas cannot be used.",
                       details={"schema": logical})
    validate_identifier(logical, "schema")
    limit = max_schema_name_length(caller.user_id)
    if len(logical) > limit:
        raise AppError(ErrorCode.INVALID_IDENTIFIER,
                       f"Schema names can be at most {limit} characters.",
                       details={"schema": logical})
    return caller.prefix + logical


def to_logical(caller: Caller, physical: str) -> str | None:
    """'u7_college' -> 'college' for user 7; None for anything the caller does not own."""
    prefix = caller.prefix
    if physical.startswith(prefix) and len(physical) > len(prefix):
        return physical[len(prefix):]
    return None


def owns(caller: Caller, physical: str) -> bool:
    return to_logical(caller, physical) is not None


def logicalize_text(caller: Caller, text: str) -> str:
    """Replace the caller's physical schema names in display text (DDL previews)."""
    return re.sub(rf"\b{re.escape(caller.prefix)}([a-z_][a-z0-9_]*)", r"\1", text)


def default_schema() -> str:
    return settings.default_schema


def table_not_found(schema: str, table: str) -> AppError:
    """Same error whether the table is missing or someone else's (it is invisible)."""
    return AppError(ErrorCode.TABLE_NOT_FOUND, f"Table {schema}.{table} was not found.",
                    details={"schema": schema, "table": table})


def schema_not_found(schema: str) -> AppError:
    return AppError(ErrorCode.SCHEMA_NOT_FOUND,
                    f"You don't have a schema named {schema!r} yet. It is created automatically "
                    "when you create or import your first table in it.",
                    details={"schema": schema})
