"""Safety layer for LLM-generated SQL. Nothing model-written reaches Postgres
without passing validate_sql().

It parses the query (sqlglot, Postgres dialect) and enforces:
  1. exactly ONE statement, and it must be a SELECT (or UNION of SELECTs)
  2. no data-changing or dangerous nodes anywhere (CTE DML, SELECT INTO, FOR UPDATE ...)
  3. no dangerous functions (pg_sleep, pg_read_file, lo_import, dblink, set_config,
     nextval ...) and no schema-qualified function calls
  4. every table is in the allow-list (no pg_catalog / information_schema / hidden tables)
  5. no sensitive columns (password, token, api_key ...)
  6. no whole-row references (to_jsonb(u), u::text, row_to_json(u.*)) that would
     smuggle sensitive columns out inside one value
  7. a LIMIT is present and never above max_rows

This is defence in depth. The real wall is the read-only Postgres role (and
column-level GRANTs that keep sensitive columns away from it).
"""
from __future__ import annotations

import re
from typing import Collection

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from app.ai.exceptions import SQLGuardError

__all__ = ["validate_sql", "SQLGuardError", "SENSITIVE_COLUMN_RE"]

MAX_SQL_CHARS = 8000

SENSITIVE_COLUMN_RE = re.compile(
    r"(password|passwd|pwd|secret|token|api[_-]?key|private[_-]?key|hash|salt|"
    r"ssn|card[_-]?number|cvv|otp)",
    re.IGNORECASE,
)

_ROOT_TYPES = (exp.Select, getattr(exp, "SetOperation", exp.Union))

_FORBIDDEN_NODE_NAMES = (
    "Insert", "Update", "Delete", "Merge", "Drop", "Create", "Alter", "AlterTable",
    "Command", "Copy", "TruncateTable", "Into", "Lock", "Set", "Use", "Transaction",
    "Commit", "Rollback", "Grant", "Revoke", "Pragma", "Analyze", "Show", "Describe",
    "Kill", "Declare",
)
_FORBIDDEN_NODES = tuple(getattr(exp, n) for n in _FORBIDDEN_NODE_NAMES if hasattr(exp, n))

_BLOCKED_FUNCS = {
    "set_config", "current_setting",
    "nextval", "setval", "currval", "lastval",
    "version", "current_version",  # sqlglot parses version() as CURRENT_VERSION
    "inet_server_addr", "inet_server_port", "inet_client_addr",
    "inet_client_port",
}
_BLOCKED_PREFIXES = ("pg_", "lo_", "dblink", "txid_")
_BLOCKED_SUFFIXES = ("_to_xml", "_to_xml_and_xmlschema")


def validate_sql(
    sql: str,
    allowed_tables: Collection[str],
    *,
    blocked_tables: Collection[str] = (),
    max_rows: int = 100,
    schema: str = "public",
) -> str:
    """Return a safe, normalized SQL string, or raise SQLGuardError.

    blocked_tables: real tables the model must not see (hidden ones). CTEs may not
    reuse these names, which would otherwise let a query alias around the allow-list.
    """
    if not sql or not sql.strip():
        raise SQLGuardError("Empty query.")
    if len(sql) > MAX_SQL_CHARS:
        raise SQLGuardError("Query is too long.")
    if "\x00" in sql:
        raise SQLGuardError("Invalid characters in query.")

    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except (SqlglotError, RecursionError) as exc:
        first_line = str(exc).splitlines()[0][:200] if str(exc) else "parse error"
        raise SQLGuardError(f"Could not parse SQL: {first_line}") from exc

    if len(statements) != 1:
        raise SQLGuardError("Exactly one SQL statement is allowed.")

    tree = statements[0]
    if not isinstance(tree, _ROOT_TYPES):
        raise SQLGuardError("Only SELECT statements are allowed.")

    schema = schema.lower()
    _check_forbidden_nodes(tree)
    _check_functions(tree)
    _check_tables(tree, {t.lower() for t in allowed_tables},
                  {t.lower() for t in blocked_tables}, schema)
    _check_columns(tree)
    _check_whole_row_refs(tree, schema)
    tree = _enforce_limit(tree, max_rows)

    return tree.sql(dialect="postgres")


# --------------------------------------------------------------------------- #
def _check_forbidden_nodes(tree: exp.Expression) -> None:
    if _FORBIDDEN_NODES:
        bad = tree.find(*_FORBIDDEN_NODES)
        if bad is not None:
            raise SQLGuardError(f"Forbidden SQL construct: {type(bad).__name__.upper()}.")


def _check_functions(tree: exp.Expression) -> None:
    for dot in tree.find_all(exp.Dot):
        if isinstance(dot.expression, exp.Func):
            raise SQLGuardError("Schema-qualified function calls are not allowed.")

    for node in tree.find_all(exp.Func):
        name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
        if (
            name in _BLOCKED_FUNCS
            or name.startswith(_BLOCKED_PREFIXES)
            or name.endswith(_BLOCKED_SUFFIXES)
        ):
            raise SQLGuardError(f"Function not allowed: {name}.")


def _check_tables(
    tree: exp.Expression, allowed: set[str], blocked: set[str], schema: str
) -> None:
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}

    reused = cte_names & blocked
    if reused:
        raise SQLGuardError(f"CTE name not allowed: {sorted(reused)[0]}.")

    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise SQLGuardError("Table functions are not allowed.")
        if table.catalog:
            raise SQLGuardError("Cross-database references are not allowed.")

        name = table.name.lower()
        table_schema = table.db.lower() if table.db else ""

        if table_schema and table_schema != schema:
            raise SQLGuardError(f"Schema not allowed: {table_schema}.")
        if not table_schema and name in cte_names:
            continue  # reference to a CTE defined in this query
        if name not in allowed:
            raise SQLGuardError(f"Table not allowed: {name}.")


def _check_columns(tree: exp.Expression) -> None:
    for col in tree.find_all(exp.Column):
        if SENSITIVE_COLUMN_RE.search(col.name or ""):
            raise SQLGuardError(f"Column not allowed: {col.name}.")


def _relation_names(tree: exp.Expression) -> set[str]:
    """Every name a row value could be referenced by: tables, aliases, CTEs, subqueries."""
    names: set[str] = set()
    for table in tree.find_all(exp.Table):
        names.add(table.name.lower())
        if table.alias:
            names.add(table.alias.lower())
    for node in tree.find_all(exp.Subquery, exp.CTE):
        if node.alias:
            names.add(node.alias.lower())
    return names


def _check_whole_row_refs(tree: exp.Expression, schema: str) -> None:
    relations = _relation_names(tree)

    for col in tree.find_all(exp.Column):
        if isinstance(col.this, exp.Star):
            continue  # t.* handled below
        qualifier = (col.table or "").lower()
        if col.name.lower() in relations and qualifier in ("", schema):
            raise SQLGuardError(
                f"Whole-row reference not allowed: {col.name}. "
                "Select named columns, qualified with the table alias."
            )

    for star in tree.find_all(exp.Star):
        parent = star.parent
        if isinstance(parent, exp.Column):  # t.*
            parent = parent.parent
        if isinstance(parent, (exp.Select, exp.Count)):
            continue  # SELECT * / SELECT t.* / COUNT(*): output columns are filtered at runtime
        raise SQLGuardError("'*' is only allowed as a select column or inside COUNT(*).")


def _enforce_limit(tree: exp.Expression, max_rows: int) -> exp.Expression:
    limit = tree.args.get("limit")  # exp.Limit, or exp.Fetch for FETCH FIRST n ROWS
    if limit is None:
        return tree.limit(max_rows)

    value = limit.expression if isinstance(limit, exp.Limit) else None
    if isinstance(value, exp.Literal) and value.is_int and int(value.name) <= max_rows:
        return tree
    return tree.limit(max_rows)  # replaces a huge, non-literal or FETCH limit
