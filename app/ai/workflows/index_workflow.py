"""Builds the RAG knowledge base in ChromaDB from the two sources of truth:

1. User uploads  -> index_document(): text/markdown/CSV/JSON are used as-is;
   PDFs, images and audio are transcribed by Gemini first. Chunks are owned by
   the uploading user.
2. PostgreSQL    -> index_table(): rows of an allowed table become one text
   chunk each ("column: value" lines, sensitive columns removed). Rows are
   owned by `owner_column` when given, otherwise shared with every user.
   Run it from a job/cron (or `python -m app.ai.workflows.index_workflow`), not
   per request.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import text as sql_text

from app.ai.agents.file_reader_agent import FileReaderAgent
from app.ai.config import settings
from app.ai.db.engine import get_read_engine
from app.ai.db.schema_inspector import get_schema
from app.ai.db.sql_guard import SENSITIVE_COLUMN_RE
from app.ai.db.vector_store import SHARED_OWNER, VectorItem, add_documents, delete_documents
from app.ai.schemas.llm_io import Usage
from app.ai.utils.chunking import split_text

log = logging.getLogger(__name__)

TEXT_TYPES = {"text/plain", "text/markdown", "text/csv", "application/json"}
_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")
MAX_ROW_CHARS = 4000


@dataclass
class IndexResult:
    doc_id: str
    chunks: int
    characters: int
    usage: Usage


def _decode(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


async def index_document(
    *,
    user_id: str,
    content: bytes | None = None,
    text: str | None = None,
    mime_type: str = "text/plain",
    filename: str = "document",
    title: str | None = None,
) -> IndexResult:
    """Add one uploaded document to the user's knowledge base."""
    if not user_id:
        raise ValueError("A user is required.")
    usage = Usage()
    if text is None:
        if not content:
            raise ValueError("The document is empty.")
        if mime_type in TEXT_TYPES:
            text = _decode(content)
        else:
            reader = FileReaderAgent()
            text = await reader.read(content, mime_type, filename)
            usage = reader.usage
    if not text or not text.strip():
        raise ValueError("No readable text found in the document.")

    chunks = split_text(text)
    if len(chunks) > settings.KB_MAX_CHUNKS_PER_DOC:
        raise ValueError(
            f"Document is too large ({len(chunks)} chunks, max {settings.KB_MAX_CHUNKS_PER_DOC})."
        )

    doc_id = uuid.uuid4().hex
    created = datetime.now(timezone.utc).isoformat()
    items = [
        VectorItem(
            content=chunk,
            owner=user_id,
            doc_id=doc_id,
            chunk=i,
            metadata={"source": "upload", "filename": filename[:255], "mime_type": mime_type,
                      "title": (title or filename)[:255], "created_at": created},
        )
        for i, chunk in enumerate(chunks)
    ]
    stored = await add_documents("documents", items)
    log.info("Indexed document %s: %d chunks", doc_id, stored)
    return IndexResult(doc_id=doc_id, chunks=stored, characters=len(text), usage=usage)


async def delete_document(*, user_id: str, doc_id: str) -> bool:
    return await delete_documents("documents", owner=user_id, doc_id=doc_id) > 0


# --------------------------------------------------------------------------- #
# PostgreSQL tables
# --------------------------------------------------------------------------- #
def _row_text(table: str, row: dict) -> str:
    lines = [f"Record from table {table}:"]
    for key, value in row.items():
        if value is None or value == "":
            continue
        lines.append(f"{key}: {value}")
    return "\n".join(lines)[:MAX_ROW_CHARS]


async def index_table(
    table: str,
    *,
    pk: str = "id",
    owner_column: str | None = None,
    batch_size: int = 500,
    max_rows: int = 50_000,
) -> int:
    """Index (or re-index) every row of an allowed table. Returns rows indexed.

    Only tables the SQL agent may read are accepted, and sensitive columns are
    dropped, so the knowledge base never exposes more than text-to-SQL would.
    """
    table = table.lower()
    schema = await get_schema()
    info = schema.tables.get(table)
    if info is None:
        raise ValueError(f"Table not available for indexing: {table}")
    columns = [c.name for c in info.columns if not SENSITIVE_COLUMN_RE.search(c.name)]
    for ident in (table, pk, *(owner_column and [owner_column] or []), *columns):
        if not _IDENT.match(ident):
            raise ValueError(f"Unsupported identifier: {ident!r}")
    if pk not in columns:
        raise ValueError(f"Primary key column {pk!r} not found in {table}.")
    if owner_column and owner_column not in columns:
        raise ValueError(f"Owner column {owner_column!r} not found in {table}.")

    db_schema = settings.DB_SCHEMA
    select_cols = ", ".join(f'"{c}"' for c in columns)
    total = 0
    last_pk = None
    while total < max_rows:
        where = f'WHERE "{pk}" > :last ' if last_pk is not None else ""
        stmt = sql_text(
            f'SELECT {select_cols} FROM "{db_schema}"."{table}" {where}'
            f'ORDER BY "{pk}" LIMIT :n'
        )
        params = {"n": batch_size, **({"last": last_pk} if last_pk is not None else {})}
        async with get_read_engine().connect() as conn:
            rows = [dict(r) for r in (await conn.execute(stmt, params)).mappings().all()]
        if not rows:
            break

        items = []
        for row in rows:
            owner = str(row.get(owner_column)) if owner_column else SHARED_OWNER
            if owner_column and row.get(owner_column) is None:
                continue  # unowned row in an owned table: skip rather than share
            body = {k: v for k, v in row.items() if k != owner_column}
            row_id = str(row[pk])
            items.append(VectorItem(
                content=_row_text(table, body),
                owner=owner,
                doc_id=f"db:{table}:{hashlib.sha1(row_id.encode()).hexdigest()[:16]}",
                metadata={"source": "database", "table": table, "row_id": row_id},
            ))
        total += await add_documents("database", items)
        last_pk = rows[-1][pk]
        if len(rows) < batch_size:
            break

    log.info("Indexed %d rows from %s", total, table)
    return total


async def index_configured_tables() -> dict[str, int]:
    """Index every table listed in RAG_INDEX_TABLES (shared ownership, pk 'id')."""
    return {t: await index_table(t) for t in sorted(settings.RAG_INDEX_TABLES)}


def _main() -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Index PostgreSQL tables into ChromaDB.")
    parser.add_argument("tables", nargs="*", help="Tables (default: RAG_INDEX_TABLES)")
    parser.add_argument("--pk", default="id")
    parser.add_argument("--owner-column", default=None)
    args = parser.parse_args()

    async def run() -> None:
        from app.ai.lifecycle import shutdown, startup

        await startup()
        try:
            if args.tables:
                for t in args.tables:
                    n = await index_table(t, pk=args.pk, owner_column=args.owner_column)
                    print(f"{t}: {n} rows")
            else:
                print(await index_configured_tables())
        finally:
            await shutdown()

    asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    _main()
