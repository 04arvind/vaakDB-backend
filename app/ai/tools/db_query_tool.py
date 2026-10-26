"""Tool: run one read-only SELECT through the full guard pipeline (db/reader.py)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.ai.db.reader import run_select
from app.ai.exceptions import QueryExecutionError, SQLGuardError
from app.ai.tools.tool_registry import Tool, ToolContext, ToolError


class QueryArgs(BaseModel):
    sql: str = Field(min_length=1, max_length=8000, description="One PostgreSQL SELECT statement.")
    max_rows: int = Field(default=50, ge=1, le=100)


async def _handle(args: BaseModel, ctx: ToolContext) -> dict[str, Any]:
    assert isinstance(args, QueryArgs)
    try:
        result = await run_select(args.sql, max_rows=args.max_rows)
    except (SQLGuardError, QueryExecutionError) as exc:
        # Safe to show the model: it lets it correct the query.
        raise ToolError(str(exc)) from exc
    return {
        "sql": result.sql,
        "columns": result.columns,
        "rows": result.rows,
        "row_count": result.row_count,
        "truncated": result.truncated,
    }


QUERY_TOOL = Tool(
    name="query_database",
    description=(
        "Run a read-only PostgreSQL SELECT against the user's database and return rows. "
        "Only tables from the provided schema are allowed."
    ),
    args_model=QueryArgs,
    handler=_handle,
)
