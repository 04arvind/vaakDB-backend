"""Write policy: what counts as destructive and what needs an explicit opt-in.

Every user may do anything in their own workspace, but destructive changes are
never implicit:
- UPDATE / DELETE need a WHERE condition, or `all_rows: true` set explicitly
  (in chat, only when the user's own message says "all"/"every"/"entire")
- replacing an existing table (DROP + CREATE) needs `if_exists: "replace"`
  (in chat, only when the user's own message says replace/overwrite)
TRUNCATE and ALTER are never generated.
"""
from __future__ import annotations

from app.core.exceptions import AppError, ErrorCode
from app.schemas.plans import AnyPlan, CreateTablePlan, DeleteRowsPlan, UpdateRowsPlan


def destructive_reasons(plan: AnyPlan, schema_label: str | None = None) -> list[str]:
    target = f"{schema_label or plan.schema_name}.{plan.table}"
    reasons = []
    if isinstance(plan, CreateTablePlan) and plan.if_exists == "replace":
        reasons.append(f"DROP and recreate {target}")
    if isinstance(plan, (UpdateRowsPlan, DeleteRowsPlan)) and not plan.where:
        verb = "UPDATE" if isinstance(plan, UpdateRowsPlan) else "DELETE"
        reasons.append(f"{verb} every row of {target} (no WHERE)")
    return reasons


def check_plan_safety(plan: AnyPlan) -> None:
    """Raises DESTRUCTIVE_OPERATION_BLOCKED for an implicit change to every row."""
    if isinstance(plan, (UpdateRowsPlan, DeleteRowsPlan)) and not plan.where and not plan.all_rows:
        raise AppError(
            ErrorCode.DESTRUCTIVE_OPERATION_BLOCKED,
            "UPDATE and DELETE need a WHERE condition saying which rows to change. "
            "To change every row, say so explicitly (for example \"delete all rows\").",
        )
