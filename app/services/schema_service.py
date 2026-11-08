"""Catalog reads on target_db (pg_catalog / information_schema), always with bound
parameters and on the read-only engine. No permission logic here: callers
(metadata_ops, plan_validator) filter the results by role."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Connection, text

from app.sql.parser import PgType, pg_type_from_catalog

_TABLE_KINDS = "('r', 'p', 'v', 'm', 'f')"
_KIND_NAMES = {"r": "table", "p": "table", "v": "view", "m": "materialized view",
               "f": "foreign table"}


@dataclass
class CatalogColumn:
    name: str
    data_type: str
    nullable: bool
    default: str | None
    position: int
    char_len: int | None = None
    precision: int | None = None
    scale: int | None = None
    primary_key: bool = False
    references: str | None = None  # "schema.table.column"

    @property
    def pg_type(self) -> PgType | None:
        return pg_type_from_catalog(self.data_type, self.char_len, self.precision, self.scale)

    @property
    def display_type(self) -> str:
        t = self.pg_type
        return str(t) if t else self.data_type


@dataclass
class CatalogTable:
    schema: str
    name: str
    kind: str
    estimated_rows: int | None
    columns: list[CatalogColumn] = field(default_factory=list)

    def column(self, name: str) -> CatalogColumn | None:
        return next((c for c in self.columns if c.name == name), None)


def existing_schemas(conn: Connection, names: list[str]) -> set[str]:
    if not names:
        return set()
    rows = conn.execute(
        text("SELECT nspname FROM pg_namespace WHERE nspname = ANY(:names)"), {"names": names}
    )
    return {r[0] for r in rows}


def owned_schemas(conn: Connection, prefix: str) -> list[str]:
    """Physical schemas whose name starts with a workspace prefix (e.g. "u7_").
    starts_with() instead of LIKE: "_" in the prefix would be a LIKE wildcard."""
    rows = conn.execute(
        text("SELECT nspname FROM pg_namespace WHERE starts_with(nspname, :prefix) "
             "ORDER BY nspname"),
        {"prefix": prefix},
    )
    return [r[0] for r in rows]


def schema_exists(conn: Connection, schema: str) -> bool:
    return schema in existing_schemas(conn, [schema])


def list_tables(conn: Connection, schema: str) -> list[CatalogTable]:
    rows = conn.execute(
        text(
            f"""
            SELECT c.relname, c.relkind, c.reltuples::bigint AS est
            FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = :schema AND c.relkind IN {_TABLE_KINDS}
            ORDER BY c.relname
            """
        ),
        {"schema": schema},
    )
    return [
        CatalogTable(schema=schema, name=r.relname, kind=_KIND_NAMES.get(r.relkind, "table"),
                     estimated_rows=int(r.est) if r.est is not None and r.est >= 0 else None)
        for r in rows
    ]


def get_table(conn: Connection, schema: str, table: str) -> CatalogTable | None:
    """Table with columns, PK and FKs, or None if it does not exist."""
    head = conn.execute(
        text(
            f"""
            SELECT c.oid, c.relkind, c.reltuples::bigint AS est
            FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = :schema AND c.relname = :table AND c.relkind IN {_TABLE_KINDS}
            """
        ),
        {"schema": schema, "table": table},
    ).first()
    if head is None:
        return None

    cols = conn.execute(
        text(
            """
            SELECT a.attname AS name,
                   format_type(a.atttypid, a.atttypmod) AS full_type,
                   t.typname AS udt,
                   NOT a.attnotnull AS nullable,
                   pg_get_expr(d.adbin, d.adrelid) AS default_expr,
                   a.attnum AS position,
                   information_schema._pg_char_max_length(a.atttypid, a.atttypmod) AS char_len,
                   information_schema._pg_numeric_precision(a.atttypid, a.atttypmod) AS num_prec,
                   information_schema._pg_numeric_scale(a.atttypid, a.atttypmod) AS num_scale
            FROM pg_attribute a
            JOIN pg_type t ON t.oid = a.atttypid
            LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
            WHERE a.attrelid = :oid AND a.attnum > 0 AND NOT a.attisdropped
            ORDER BY a.attnum
            """
        ),
        {"oid": head.oid},
    ).all()

    keys = conn.execute(
        text(
            """
            SELECT c.contype, a.attname AS col,
                   rn.nspname AS ref_schema, rc.relname AS ref_table, ra.attname AS ref_col
            FROM pg_constraint c
            CROSS JOIN LATERAL unnest(c.conkey, COALESCE(c.confkey, c.conkey))
                 AS k(attnum, refattnum)
            JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum
            LEFT JOIN pg_class rc ON rc.oid = c.confrelid
            LEFT JOIN pg_namespace rn ON rn.oid = rc.relnamespace
            LEFT JOIN pg_attribute ra ON ra.attrelid = c.confrelid AND ra.attnum = k.refattnum
            WHERE c.conrelid = :oid AND c.contype IN ('p', 'f')
            """
        ),
        {"oid": head.oid},
    ).all()
    pks = {k.col for k in keys if k.contype == "p"}
    fks = {k.col: f"{k.ref_schema}.{k.ref_table}.{k.ref_col}" for k in keys if k.contype == "f"}

    result = CatalogTable(
        schema=schema, name=table, kind=_KIND_NAMES.get(head.relkind, "table"),
        estimated_rows=int(head.est) if head.est is not None and head.est >= 0 else None,
    )
    for c in cols:
        result.columns.append(
            CatalogColumn(
                name=c.name,
                data_type=_information_schema_type(c.full_type, c.udt),
                nullable=bool(c.nullable),
                default=c.default_expr,
                position=int(c.position),
                char_len=c.char_len,
                precision=c.num_prec,
                scale=c.num_scale,
                primary_key=c.name in pks,
                references=fks.get(c.name),
            )
        )
    return result


def _information_schema_type(full_type: str, udt: str) -> str:
    """'character varying(40)' -> 'character varying' (length is kept separately)."""
    base = full_type.split("(", 1)[0].strip()
    return base if base else udt


def jsonable_row(row: Any) -> dict[str, Any]:
    from app.sql.validator import to_jsonable

    return {k: to_jsonable(v) for k, v in dict(row).items()}
