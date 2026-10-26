"""Tool: insert records into a REGISTERED write target (db/writer.py).

The model never writes SQL here: it names a target and supplies field values,
which are validated against the target's Pydantic model. Ownership comes from
ToolContext.user_id, never from the arguments. The registry only runs this tool
when the caller passes allow_writes=True.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.ai.db.writer import get_target, insert_records
from app.ai.exceptions import UnknownTargetError
from app.ai.tools.tool_registry import Tool, ToolContext, ToolError


class WriteArgs(BaseModel):
    target: str = Field(pattern=r"^[a-z_][a-z0-9_]{0,62}$")
    records: list[dict[str, Any]] = Field(min_length=1, max_length=100)


async def _handle(args: BaseModel, ctx: ToolContext) -> dict[str, Any]:
    assert isinstance(args, WriteArgs)
    try:
        target = get_target(args.target)
    except UnknownTargetError as exc:
        raise ToolError(str(exc)) from exc

    clean = []
    for rec in args.records:
        rec = {k: v for k, v in rec.items() if k != target.owner_column}  # never trust ownership
        try:
            clean.append(target.model.model_validate(rec))
        except ValueError as exc:
            raise ToolError(f"Record failed validation for '{target.name}'.") from exc

    if ctx.dry_run:
        return {"inserted": 0, "ids": [], "preview": [c.model_dump(mode="json") for c in clean]}
    try:
        ids = await insert_records(target.name, clean, user_id=ctx.user_id)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return {"inserted": len(ids), "ids": ids}


WRITE_TOOL = Tool(
    name="insert_records",
    description="Insert validated records into a registered table (e.g. expenses).",
    args_model=WriteArgs,
    handler=_handle,
    writes=True,
)
