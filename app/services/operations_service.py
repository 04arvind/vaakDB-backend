"""Pending operations: two-step execution.

plan  -> create_pending()  status=pending, preview stored, expires after OPERATION_TTL_MINUTES
user  -> confirm()         same user only; re-validates against the live database and
                           only the caller's own workspace, then executes in one transaction
      -> cancel()

confirm() is idempotent: the operation row is locked (SELECT ... FOR UPDATE)
and moved to 'confirmed' before executing, so a second confirm either gets
the stored result (executed), OPERATION_IN_PROGRESS, or the stored failure.
"""
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import AppError, ErrorCode
from app.db.models import Operation
from app.db.target_db import get_read_engine
from app.schemas.plans import AnyPlan, dump_plan, parse_plan
from app.services.audit_service import audit
from app.services.metadata_ops import Caller
from app.services.plan_executor import AfterBatch, execute_plan
from app.services.plan_validator import ValidatedPlan, validate_plan

log = logging.getLogger(__name__)

# A 'confirmed' operation older than this was interrupted (process killed / crashed).
_STALE_AFTER = timedelta(milliseconds=settings.statement_timeout_ms) + timedelta(minutes=2)


def now() -> datetime:
    return datetime.now(UTC)


def _row_count(v: ValidatedPlan) -> int | None:
    return v.affected_rows if v.affected_rows is not None else v.rows_to_write


def create_pending(
    db: Session,
    *,
    caller: Caller,
    validated: ValidatedPlan,
    conversation_id: uuid.UUID | None = None,
    upload_id: uuid.UUID | None = None,
) -> Operation:
    plan = validated.plan
    op_id = uuid.uuid4()
    expires = now() + timedelta(minutes=settings.operation_ttl_minutes)
    preview = validated.preview()
    preview["operation_id"] = str(op_id)
    preview["expires_at"] = expires.isoformat()
    op = Operation(
        id=op_id,
        user_id=caller.user_id,
        conversation_id=conversation_id,
        upload_id=upload_id,
        type=plan.kind,
        status="pending",
        target_schema=validated.logical_schema or plan.schema_name,
        target_table=plan.table,
        plan_json=dump_plan(validated.source_plan or plan),  # logical names
        preview_json=preview,
        expires_at=expires,
    )
    db.add(op)
    db.commit()
    audit("plan", user_id=caller.user_id, status="pending", operation_id=op_id,
          upload_id=upload_id, operation_type=plan.kind,
          target_schema=validated.logical_schema or plan.schema_name,
          target_table=plan.table, row_count=_row_count(validated),
          details={"mode": preview.get("mode"), "failed_rows": validated.cast.failed_row_count,
                   "destructive": preview.get("destructive")})
    return op


def _load_for_user(db: Session, op_id: uuid.UUID, caller: Caller, *, lock: bool) -> Operation:
    stmt = select(Operation).where(Operation.id == op_id)
    if lock:
        stmt = stmt.with_for_update()
    op = db.execute(stmt).scalar_one_or_none()
    if op is None or op.user_id != caller.user_id:  # other users' operations look missing
        raise AppError(ErrorCode.OPERATION_NOT_FOUND)
    return op


def _refresh_state(db: Session, op: Operation) -> None:
    """Lazy expiry + recovery of interrupted executions."""
    if op.status == "pending" and now() >= op.expires_at:
        op.status = "expired"
        db.commit()
        audit("expire", user_id=op.user_id, status="expired", operation_id=op.id,
              operation_type=op.type, target_schema=op.target_schema, target_table=op.target_table)
    elif op.status == "confirmed" and op.confirmed_at and now() - op.confirmed_at > _STALE_AFTER:
        op.status = "failed"
        op.error = ("The operation was interrupted before it finished. Its transaction was "
                    "rolled back by the database; please check the table and try again.")
        db.commit()
        audit("execute", user_id=op.user_id, status="failed", operation_id=op.id,
              operation_type=op.type, target_schema=op.target_schema,
              target_table=op.target_table, error="interrupted")


def _iso(moment: datetime | None) -> str | None:
    """Always UTC, whatever the database session's TimeZone is."""
    return moment.astimezone(UTC).isoformat() if moment else None


def operation_view(op: Operation) -> dict[str, Any]:
    return {
        "operation_id": str(op.id),
        "type": op.type,
        "status": op.status,
        "target": {"schema": op.target_schema, "table": op.target_table},
        "preview": op.preview_json,
        "result": op.result_json,
        "error": op.error,
        "created_at": _iso(op.created_at),
        "expires_at": _iso(op.expires_at),
        "confirmed_at": _iso(op.confirmed_at),
        "executed_at": _iso(op.executed_at),
    }


def get_operation(db: Session, op_id: uuid.UUID, caller: Caller) -> dict[str, Any]:
    op = _load_for_user(db, op_id, caller, lock=False)
    _refresh_state(db, op)
    return operation_view(op)


def cancel(db: Session, op_id: uuid.UUID, caller: Caller) -> dict[str, Any]:
    op = _load_for_user(db, op_id, caller, lock=True)
    _refresh_state(db, op)
    if op.status == "cancelled":
        db.commit()
        return operation_view(op)
    if op.status != "pending":
        db.commit()
        raise AppError(ErrorCode.OPERATION_NOT_PENDING,
                       f"Only pending operations can be cancelled (this one is {op.status}).",
                       details={"status": op.status})
    op.status = "cancelled"
    db.commit()
    audit("cancel", user_id=caller.user_id, status="cancelled", operation_id=op.id,
          operation_type=op.type, target_schema=op.target_schema, target_table=op.target_table)
    return operation_view(op)


def _result_envelope(op: Operation) -> dict[str, Any]:
    result = op.result_json or {}
    written = result.get("rows_written")
    if written is not None:
        msg = f"Done: wrote {written} row(s) to {op.target_schema}.{op.target_table}."
        if result.get("table_created"):
            msg = f"Done: created {op.target_schema}.{op.target_table} and inserted {written} row(s)."
    elif "rows_updated" in result:
        msg = f"Done: updated {result['rows_updated']} row(s) in {op.target_schema}.{op.target_table}."
    elif "rows_deleted" in result:
        msg = f"Done: deleted {result['rows_deleted']} row(s) from {op.target_schema}.{op.target_table}."
    else:
        msg = "Done."
    if result.get("rows_skipped"):
        msg += f" {result['rows_skipped']} row(s) were skipped because they did not match the column types."
    return {"type": "operation_result", "data": operation_view(op), "message": msg}


def confirm(db: Session, op_id: uuid.UUID, caller: Caller, *,
            after_batch: AfterBatch | None = None) -> dict[str, Any]:
    op = _load_for_user(db, op_id, caller, lock=True)
    _refresh_state(db, op)
    if op.status == "executed":  # idempotent: return the stored result, run nothing
        db.commit()
        return _result_envelope(op)
    if op.status == "confirmed":
        db.commit()
        raise AppError(ErrorCode.OPERATION_IN_PROGRESS, "This operation is already running.")
    if op.status == "expired":
        db.commit()
        raise AppError(ErrorCode.OPERATION_EXPIRED)
    if op.status != "pending":
        db.commit()
        raise AppError(ErrorCode.OPERATION_NOT_PENDING,
                       f"This operation is {op.status} and cannot be confirmed."
                       + (f" Error: {op.error}" if op.error else ""),
                       details={"status": op.status})

    op.status = "confirmed"
    op.confirmed_at = now()
    db.commit()  # releases the row lock; concurrent confirms now see 'confirmed'
    audit("confirm", user_id=caller.user_id, status="confirmed", operation_id=op.id,
          operation_type=op.type, target_schema=op.target_schema, target_table=op.target_table)

    try:
        plan: AnyPlan = parse_plan(op.plan_json)
        with get_read_engine().connect() as conn:  # re-check against the live database
            validated = validate_plan(plan, caller=caller, conn=conn)
        result = execute_plan(validated, after_batch=after_batch)
    except Exception as exc:
        err = exc if isinstance(exc, AppError) else AppError(ErrorCode.EXECUTION_FAILED)
        if not isinstance(exc, AppError):
            log.exception("Operation %s failed unexpectedly", op.id)
        op.status = "failed"
        op.error = err.message
        op.result_json = {"error": err.to_dict()}
        op.executed_at = now()
        db.commit()
        audit("execute", user_id=caller.user_id, status="failed", operation_id=op.id,
              operation_type=op.type, target_schema=op.target_schema,
              target_table=op.target_table, error=f"{err.code}: {err.message}")
        raise err from None

    op.status = "executed"
    op.result_json = result
    op.executed_at = now()
    db.commit()
    rows = result.get("rows_written", result.get("rows_updated", result.get("rows_deleted")))
    audit("execute", user_id=caller.user_id, status="executed", operation_id=op.id,
          operation_type=op.type, target_schema=op.target_schema, target_table=op.target_table,
          row_count=rows, details={"rows_skipped": result.get("rows_skipped")})
    return _result_envelope(op)
