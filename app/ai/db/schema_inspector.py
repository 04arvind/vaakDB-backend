"""Reads the database structure so the LLM knows what exists.

- Uses the READ-ONLY engine. information_schema only lists columns the role may
  access, so column-level GRANTs automatically hide data from the model.
- Hidden tables (internal ai_* tables, alembic_version, your own list) and
  sensitive columns are never shown and never allowed in queries.
- Result is cached (default 5 minutes). Call invalidate_schema_cache() after migrations.
- Large schemas are narrowed to the tables relevant to the question (plus the
  tables they reference) so the prompt stays inside SCHEMA_MAX_TOKENS.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Iterable

from sqlalchemy import text

from app.ai.db.engine import cfg, get_read_engine
from app.ai.db.sql_guard import SENSITIVE_COLUMN_RE
from app.ai.utils.token_counter import estimate_tokens

DEFAULT_HIDDEN_TABLES = {"alembic_version"}
DEFAULT_HIDDEN_PREFIXES = ("ai_",)

_COLUMNS_SQL = """
SELECT table_name, column_name, data_type, udt_name, is_nullable
FROM information_schema.columns
WHERE table_schema = :schema
ORDER BY table_name, ordinal_position
"""

# PK/FK come from pg_catalog on purpose: information_schema.table_constraints and
# constraint_column_usage hide constraints from roles that only have SELECT on a table
# (exactly how the read-only AI role is set up), which would silently drop all keys.
_PK_SQL = """
SELECT cl.relname AS table_name, a.attname AS column_name
FROM pg_constraint c
JOIN pg_class cl ON cl.oid = c.conrelid
JOIN pg_namespace n ON n.oid = cl.relnamespace
CROSS JOIN LATERAL unnest(c.conkey) AS k(attnum)
JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum
WHERE c.contype = 'p' AND n.nspname = :schema
"""

_FK_SQL = """
SELECT cl.relname AS table_name, a.attname AS column_name,
       rcl.relname AS ref_table, ra.attname AS ref_column
FROM pg_constraint c
JOIN pg_class cl ON cl.oid = c.conrelid
JOIN pg_namespace n ON n.oid = cl.relnamespace
JOIN pg_class rcl ON rcl.oid = c.confrelid
CROSS JOIN LATERAL unnest(c.conkey, c.confkey) AS k(attnum, refattnum)
JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum
JOIN pg_attribute ra ON ra.attrelid = c.confrelid AND ra.attnum = k.refattnum
WHERE c.contype = 'f' AND n.nspname = :schema
"""

_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool
    primary_key: bool = False
    references: str | None = None  # "table.column"


@dataclass
class TableInfo:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)


@dataclass
class SchemaInfo:
    tables: dict[str, TableInfo]
    hidden: set[str]  # real tables the model must not see
    loaded_at: float

    def render(self, only: Iterable[str] | None = None) -> str:
        """Compact text description for the system prompt."""
        wanted = {t.lower() for t in only} if only is not None else None
        lines: list[str] = []
        for name in sorted(self.tables):
            if wanted is not None and name not in wanted:
                continue
            lines.append(f"TABLE {name}")
            for col in self.tables[name].columns:
                flags = []
                if col.primary_key:
                    flags.append("PK")
                if col.references:
                    flags.append(f"FK -> {col.references}")
                if col.nullable:
                    flags.append("nullable")
                suffix = f" [{', '.join(flags)}]" if flags else ""
                lines.append(f"  - {col.name}: {col.data_type}{suffix}")
        return "\n".join(lines)

    def relevant_tables(self, question: str, *, limit: int = 12) -> list[str]:
        """Tables whose names/columns share words with the question, plus FK neighbours."""
        words = _words(question)
        scored: list[tuple[int, str]] = []
        for name, info in self.tables.items():
            score = 3 * len(words & _words(name))
            score += sum(1 for c in info.columns if words & _words(c.name))
            if score:
                scored.append((score, name))
        scored.sort(key=lambda s: (-s[0], s[1]))
        picked = [name for _, name in scored[:limit]]

        for name in list(picked):  # pull in referenced tables so JOINs are possible
            for col in self.tables[name].columns:
                ref = col.references.split(".", 1)[0] if col.references else None
                if ref and ref in self.tables and ref not in picked:
                    picked.append(ref)
        return picked


def _words(value: str) -> set[str]:
    out: set[str] = set()
    for w in _WORD.findall(value.lower().replace("_", " ")):
        if len(w) < 3:
            continue
        out.add(w)
        if w.endswith("ies") and len(w) > 4:
            out.add(w[:-3] + "y")
        elif w.endswith("s") and len(w) > 3:
            out.add(w[:-1])
        else:
            out.add(w + "s")
    return out


_cache: SchemaInfo | None = None
_lock = asyncio.Lock()


def invalidate_schema_cache() -> None:
    global _cache
    _cache = None


async def get_schema(force: bool = False) -> SchemaInfo:
    global _cache
    ttl = int(cfg("SCHEMA_CACHE_TTL", 300))
    if not force and _cache and time.time() - _cache.loaded_at < ttl:
        return _cache
    async with _lock:
        if not force and _cache and time.time() - _cache.loaded_at < ttl:
            return _cache
        _cache = await _load()
        return _cache


async def get_schema_context(
    only: Iterable[str] | None = None, *, question: str | None = None
) -> str:
    """Schema text to inject into the SQL agent prompt.

    With `question`, a schema larger than SCHEMA_MAX_TOKENS is narrowed to the
    relevant tables. Falls back to the full schema when nothing matches.
    """
    schema = await get_schema()
    full = schema.render(only)
    max_tokens = int(cfg("SCHEMA_MAX_TOKENS", 3000))
    if only is not None or not question or estimate_tokens(full) <= max_tokens:
        return full
    picked = schema.relevant_tables(question)
    return schema.render(picked) if picked else full


async def get_table_names() -> list[str]:
    return sorted((await get_schema()).tables)


async def get_allowed_tables() -> set[str]:
    return set((await get_schema()).tables)


async def get_hidden_tables() -> set[str]:
    return set((await get_schema()).hidden)


async def _load() -> SchemaInfo:
    schema = cfg("DB_SCHEMA", "public")
    hidden_cfg = {t.lower() for t in (cfg("AI_HIDDEN_TABLES") or set())}
    hidden_names = DEFAULT_HIDDEN_TABLES | hidden_cfg
    prefixes = tuple(cfg("AI_HIDDEN_PREFIXES", DEFAULT_HIDDEN_PREFIXES))
    readable = {t.lower() for t in (cfg("AI_READABLE_TABLES") or set())}

    params = {"schema": schema}
    async with get_read_engine().connect() as conn:
        col_rows = (await conn.execute(text(_COLUMNS_SQL), params)).all()
        pk_rows = (await conn.execute(text(_PK_SQL), params)).all()
        fk_rows = (await conn.execute(text(_FK_SQL), params)).all()

    pks = {(r.table_name.lower(), r.column_name.lower()) for r in pk_rows}
    fks = {
        (r.table_name.lower(), r.column_name.lower()): f"{r.ref_table}.{r.ref_column}"
        for r in fk_rows
    }

    tables: dict[str, TableInfo] = {}
    hidden: set[str] = set()

    for row in col_rows:
        table = row.table_name.lower()
        column = row.column_name.lower()

        is_hidden = (
            table in hidden_names
            or (bool(prefixes) and table.startswith(prefixes))
            or (bool(readable) and table not in readable)
        )
        if is_hidden:
            hidden.add(table)
            continue
        if SENSITIVE_COLUMN_RE.search(column):
            continue  # never describe sensitive columns to the model

        data_type = row.data_type
        if data_type == "USER-DEFINED":  # enums, pgvector, PostGIS ...
            data_type = row.udt_name
        elif data_type == "ARRAY":
            data_type = f"{row.udt_name.lstrip('_')}[]"

        info = tables.setdefault(table, TableInfo(name=table))
        info.columns.append(
            ColumnInfo(
                name=column,
                data_type=data_type,
                nullable=row.is_nullable == "YES",
                primary_key=(table, column) in pks,
                references=fks.get((table, column)),
            )
        )

    # Never point the model at a table it is not allowed to use.
    for info in tables.values():
        for col in info.columns:
            if col.references and col.references.split(".", 1)[0].lower() not in tables:
                col.references = None

    return SchemaInfo(tables=tables, hidden=hidden, loaded_at=time.time())
