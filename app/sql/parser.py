"""Identifier hygiene and the column-type whitelist.

Every schema/table/column name that reaches SQL must match
    ^[a-z_][a-z0-9_]{0,62}$
after normalization (lowercase, spaces -> underscores, other symbols dropped),
and must not be a Postgres reserved word or system column.

Allowed column types (canonical spelling on the right):
    text, varchar(n), integer, bigint, numeric(p,s), boolean, date, timestamp, uuid
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from app.core.exceptions import AppError, ErrorCode

IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
MAX_IDENTIFIER_LEN = 63

# Postgres reserved key words (cannot be used as names without quoting) plus the
# "reserved (can be function or type)" class, which also breaks unquoted DDL.
RESERVED_WORDS = frozenset({
    "all", "analyse", "analyze", "and", "any", "array", "as", "asc", "asymmetric",
    "authorization", "binary", "both", "case", "cast", "check", "collate", "collation",
    "column", "concurrently", "constraint", "create", "cross", "current_catalog",
    "current_date", "current_role", "current_schema", "current_time", "current_timestamp",
    "current_user", "default", "deferrable", "desc", "distinct", "do", "else", "end", "except",
    "false", "fetch", "for", "foreign", "freeze", "from", "full", "grant", "group", "having",
    "ilike", "in", "initially", "inner", "intersect", "into", "is", "isnull", "join", "lateral",
    "leading", "left", "like", "limit", "localtime", "localtimestamp", "natural", "not",
    "notnull", "null", "offset", "on", "only", "or", "order", "outer", "overlaps", "placing",
    "primary", "references", "returning", "right", "select", "session_user", "similar", "some",
    "symmetric", "system_user", "table", "tablesample", "then", "to", "trailing", "true",
    "union", "unique", "user", "using", "variadic", "verbose", "when", "where", "window",
    "with",
})

# Hidden system columns every Postgres table already has.
SYSTEM_COLUMNS = frozenset({"oid", "tableoid", "xmin", "xmax", "cmin", "cmax", "ctid"})

IdentKind = Literal["schema", "table", "column"]
_SUFFIX = {"schema": "_schema", "table": "_tbl", "column": "_col"}


def _invalid(message: str, value: object) -> AppError:
    return AppError(ErrorCode.INVALID_IDENTIFIER, message, details={"value": str(value)[:100]})


def is_valid_identifier(name: str, kind: IdentKind = "column") -> bool:
    if not isinstance(name, str) or not IDENTIFIER_RE.match(name):
        return False
    if name in RESERVED_WORDS:
        return False
    if kind == "column" and name in SYSTEM_COLUMNS:
        return False
    return not (kind == "schema" and name.startswith("pg_"))


def validate_identifier(name: str, kind: IdentKind = "column") -> str:
    """Strict check: the name must already be in normalized form. Raises INVALID_IDENTIFIER."""
    if not isinstance(name, str) or not IDENTIFIER_RE.match(name):
        raise _invalid(
            f"Invalid {kind} name {str(name)[:63]!r}: use lowercase letters, digits and "
            "underscores, starting with a letter or underscore (max 63 characters).",
            name,
        )
    if name in RESERVED_WORDS or (kind == "column" and name in SYSTEM_COLUMNS):
        raise _invalid(f"{name!r} is a reserved word and cannot be used as a {kind} name.", name)
    if kind == "schema" and name.startswith("pg_"):
        raise _invalid("Schema names starting with 'pg_' are reserved.", name)
    return name


def normalize_identifier(
    raw: object,
    kind: IdentKind = "column",
    *,
    on_reserved: Literal["reject", "suffix"] = "reject",
) -> str:
    """'Roll No.' -> 'roll_no', 'Date of Birth' -> 'date_of_birth', '2024 Marks' -> '_2024_marks'.

    Reserved words are rejected, or suffixed ('order' -> 'order_col') when
    on_reserved='suffix' (used for headers that come from files).
    """
    if raw is None:
        raise _invalid(f"Missing {kind} name.", raw)
    text = unicodedata.normalize("NFKD", str(raw)).encode("ascii", "ignore").decode()
    text = text.strip().lower()
    text = text.replace("%", " pct ").replace("#", " no ").replace("&", " and ")
    text = re.sub(r"[\s\-./\\:,;|+]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        raise _invalid(f"{kind.capitalize()} name {str(raw)[:63]!r} has no usable characters.", raw)
    if text[0].isdigit():
        text = "_" + text
    text = text[:MAX_IDENTIFIER_LEN].rstrip("_") or "_"

    reserved = text in RESERVED_WORDS or (kind == "column" and text in SYSTEM_COLUMNS) or (
        kind == "schema" and text.startswith("pg_")
    )
    if reserved:
        if on_reserved == "reject" or kind == "schema":
            raise _invalid(f"{text!r} is reserved and cannot be used as a {kind} name.", raw)
        text = (text[: MAX_IDENTIFIER_LEN - len(_SUFFIX[kind])] + _SUFFIX[kind])
    return validate_identifier(text, kind)


def dedupe_identifiers(names: list[str]) -> list[str]:
    """['name', 'name', 'name_2'] -> ['name', 'name_3', 'name_2'] (stable, unique)."""
    taken = set(names)
    used: set[str] = set()
    out: list[str] = []
    for name in names:
        candidate, n = name, 2
        while candidate in used or (candidate != name and candidate in taken):
            suffix = f"_{n}"
            candidate = name[: MAX_IDENTIFIER_LEN - len(suffix)] + suffix
            n += 1
        used.add(candidate)
        out.append(candidate)
    return out


# --------------------------------------------------------------------------- #
# column types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PgType:
    name: Literal["text", "varchar", "integer", "bigint", "numeric", "boolean", "date",
                  "timestamp", "uuid"]
    length: int | None = None  # varchar(n)
    precision: int | None = None  # numeric(p,s)
    scale: int | None = None

    def __str__(self) -> str:
        if self.name == "varchar":
            return f"varchar({self.length})"
        if self.name == "numeric" and self.precision is not None:
            return f"numeric({self.precision},{self.scale or 0})"
        return self.name


_SIMPLE = {
    "text": "text", "string": "text", "str": "text", "char": "text",
    "integer": "integer", "int": "integer", "int4": "integer", "smallint": "integer",
    "int2": "integer", "serial": "integer",
    "bigint": "bigint", "int8": "bigint", "bigserial": "bigint", "long": "bigint",
    "numeric": "numeric", "decimal": "numeric", "float": "numeric", "double": "numeric",
    "double precision": "numeric", "real": "numeric", "float8": "numeric", "float4": "numeric",
    "number": "numeric", "money": "numeric",
    "boolean": "boolean", "bool": "boolean",
    "date": "date",
    "timestamp": "timestamp", "timestamp without time zone": "timestamp", "datetime": "timestamp",
    "uuid": "uuid",
}
_VARCHAR = re.compile(r"^(?:varchar|character varying|char|character)\s*\(\s*(\d{1,5})\s*\)$")
_NUMERIC = re.compile(r"^(?:numeric|decimal)\s*\(\s*(\d{1,4})\s*(?:,\s*(\d{1,4})\s*)?\)$")
ALLOWED_TYPES_HELP = (
    "text, varchar(n), integer, bigint, numeric(p,s), boolean, date, timestamp, uuid"
)


def parse_pg_type(raw: object) -> PgType:
    """Parse and canonicalize a column type from the whitelist. Raises PLAN_INVALID."""
    text = re.sub(r"\s+", " ", str(raw or "").strip().lower())
    if text in ("varchar", "character varying"):
        return PgType("varchar", length=255)
    if text in _SIMPLE:
        return PgType(_SIMPLE[text])  # type: ignore[arg-type]
    m = _VARCHAR.match(text)
    if m:
        length = int(m.group(1))
        if not 1 <= length <= 10_485_760:
            raise AppError(ErrorCode.PLAN_INVALID, f"varchar length {length} is out of range.")
        return PgType("varchar", length=length)
    m = _NUMERIC.match(text)
    if m:
        precision, scale = int(m.group(1)), int(m.group(2) or 0)
        if not (1 <= precision <= 1000 and 0 <= scale <= precision):
            raise AppError(ErrorCode.PLAN_INVALID, f"numeric({precision},{scale}) is out of range.")
        return PgType("numeric", precision=precision, scale=scale)
    raise AppError(
        ErrorCode.PLAN_INVALID,
        f"Column type {str(raw)[:40]!r} is not allowed. Use one of: {ALLOWED_TYPES_HELP}.",
    )


def canonical_type(raw: object) -> str:
    return str(parse_pg_type(raw))


def pg_type_from_catalog(data_type: str, char_len: int | None, precision: int | None,
                         scale: int | None) -> PgType | None:
    """Map an existing column (information_schema) onto our types, or None if the
    column has a type we do not write to (json, arrays, ...)."""
    dt = (data_type or "").lower()
    if dt in ("character varying", "character"):
        return PgType("varchar", length=char_len or 10_485_760)
    if dt == "numeric":
        return PgType("numeric", precision=precision, scale=scale) if precision else PgType("numeric")
    if dt in ("smallint", "integer"):
        return PgType("integer")
    if dt in ("real", "double precision"):
        return PgType("numeric")
    if dt.startswith("timestamp"):
        return PgType("timestamp")
    mapping = {"text": "text", "bigint": "bigint", "boolean": "boolean", "date": "date",
               "uuid": "uuid"}
    return PgType(mapping[dt]) if dt in mapping else None  # type: ignore[arg-type]
