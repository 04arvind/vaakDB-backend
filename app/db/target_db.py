"""Engines for target_db (the user's data).

Two engines, two Postgres roles:
- read engine  -> vaak_ro: every session is read-only and has a short statement_timeout
- write engine -> vaak_rw: CREATE/INSERT/UPDATE/DELETE on ALLOWED_SCHEMAS only

LLM-written SQL (natural-language SELECT) never runs here through the write
engine. Writes only happen in plan_executor, from SQL built by SQLAlchemy Core.
"""
from __future__ import annotations

import threading

from sqlalchemy import Engine, create_engine, text

from app.core.config import settings
from app.core.exceptions import AppError, ErrorCode

APPLICATION_NAME = "vaakdb-backend"

_lock = threading.Lock()
_engines: dict[str, Engine] = {}


def _options(**values: str | int) -> str:
    return " ".join(f"-c {k}={v}" for k, v in values.items())


def _build(kind: str) -> Engine:
    if kind == "ro":
        url = settings.target_ro_url
        options = _options(
            application_name=f"{APPLICATION_NAME}-ro",
            default_transaction_read_only="on",
            statement_timeout=settings.read_timeout_ms,
            idle_in_transaction_session_timeout=15000,
        )
        pool_size = 5
    else:
        url = settings.target_rw_url
        options = _options(
            application_name=f"{APPLICATION_NAME}-rw",
            statement_timeout=settings.statement_timeout_ms,
            lock_timeout=settings.lock_timeout_ms,
            idle_in_transaction_session_timeout=60000,
        )
        pool_size = 2
    if not url:
        raise AppError(ErrorCode.NOT_CONFIGURED, "The target database is not configured.")
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=pool_size,
        max_overflow=2,
        pool_recycle=1800,
        connect_args={"options": options, "connect_timeout": 5},
    )


def _get(kind: str) -> Engine:
    engine = _engines.get(kind)
    if engine is None:
        with _lock:
            engine = _engines.get(kind)
            if engine is None:
                engine = _engines[kind] = _build(kind)
    return engine


def get_read_engine() -> Engine:
    return _get("ro")


def get_write_engine() -> Engine:
    return _get("rw")


def set_engines(read: Engine | None = None, write: Engine | None = None) -> None:
    """Override engines (tests)."""
    with _lock:
        if read is not None:
            _engines["ro"] = read
        if write is not None:
            _engines["rw"] = write


def dispose_engines() -> None:
    with _lock:
        for engine in _engines.values():
            engine.dispose()
        _engines.clear()


def ping(kind: str) -> dict[str, object]:
    """Health info for one engine: reachable, connected role (never the URL)."""
    try:
        engine = get_read_engine() if kind == "ro" else get_write_engine()
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT current_user AS role, current_database() AS db, "
                     "(SELECT rolsuper FROM pg_roles WHERE rolname = current_user) AS superuser")
            ).one()
        return {"ok": True, "role": row.role, "database": row.db, "superuser": bool(row.superuser)}
    except AppError:
        return {"ok": False, "error": "not configured"}
    except Exception:
        return {"ok": False, "error": "unreachable"}
