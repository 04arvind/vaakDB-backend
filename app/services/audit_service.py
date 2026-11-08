"""Audit trail (append-only audit_logs in meta_db).

One row per upload, plan, confirm, cancel, execution, expiry and denial. Rows
are written in their own transaction so they survive a rollback of the work
they describe. The table has a trigger that rejects UPDATE/DELETE/TRUNCATE.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from app.core.logging import scrub
from app.db.app_db import independent_session
from app.db.models import AuditLog

log = logging.getLogger(__name__)

EVENTS = ("upload", "plan", "confirm", "cancel", "execute", "expire", "denied", "read")


def audit(
    event: str,
    *,
    user_id: int | None,
    status: str,
    operation_id: uuid.UUID | None = None,
    upload_id: uuid.UUID | None = None,
    operation_type: str | None = None,
    target_schema: str | None = None,
    target_table: str | None = None,
    row_count: int | None = None,
    error: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Never raises: a failing audit write is logged loudly instead of breaking the request."""
    try:
        with independent_session() as db:
            db.add(
                AuditLog(
                    user_id=user_id,
                    event=event,
                    operation_id=operation_id,
                    upload_id=upload_id,
                    operation_type=operation_type,
                    target_schema=target_schema,
                    target_table=target_table,
                    row_count=row_count,
                    status=status,
                    error=scrub(error)[:2000] if error else None,
                    details=details,
                )
            )
    except Exception:
        log.exception("AUDIT WRITE FAILED (event=%s, operation=%s)", event, operation_id)
