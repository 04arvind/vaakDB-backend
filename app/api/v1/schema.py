"""Schema browser for the caller's own workspace: schemas, tables, table
description, rows, CSV export. No LLM involved. Schema names are logical
(the user's own names); other users' schemas can't be addressed."""
from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Path, Query
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentCaller
from app.services import metadata_ops

router = APIRouter(prefix="/schemas", tags=["Schemas"])

Name = Annotated[str, Path(pattern=r"^[a-z_][a-z0-9_]{0,62}$")]


@router.get("")
def list_schemas(caller: CurrentCaller) -> dict[str, Any]:
    return metadata_ops.list_schemas(caller)


@router.get("/{schema}/tables")
def list_tables(schema: Name, caller: CurrentCaller) -> dict[str, Any]:
    return metadata_ops.list_tables(caller, schema)


@router.get("/{schema}/tables/{table}")
def describe_table(schema: Name, table: Name, caller: CurrentCaller) -> dict[str, Any]:
    return metadata_ops.describe_table(caller, schema, table)


@router.get("/{schema}/tables/{table}/rows")
def table_rows(
    schema: Name,
    table: Name,
    caller: CurrentCaller,
    page: Annotated[int, Query(ge=1, le=1_000_000)] = 1,
    page_size: Annotated[int, Query(ge=1, le=5000)] = 50,
    sort: Annotated[str | None, Query(max_length=63)] = None,
    dir: Annotated[Literal["asc", "desc"], Query()] = "asc",
) -> dict[str, Any]:
    """Paginated rows. `sort` must be one of the table's columns."""
    return metadata_ops.table_rows(caller, schema, table, page=page, page_size=page_size,
                                   sort=sort, direction=dir)


@router.get("/{schema}/tables/{table}/csv")
def export_csv(schema: Name, table: Name, caller: CurrentCaller) -> StreamingResponse:
    """Streamed CSV, capped at CSV_EXPORT_MAX_ROWS rows."""
    chunks = metadata_ops.export_csv(caller, schema, table)  # checks run before streaming starts
    return StreamingResponse(
        chunks,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{schema}.{table}.csv"'},
    )
