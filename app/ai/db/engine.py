"""Async SQLAlchemy engines: one READ-ONLY, one RESTRICTED-WRITE.

Two engines, two Postgres roles. LLM-generated SQL only ever touches the
read engine, so even a guard bypass cannot modify data.

Drivers: plain postgresql:// URLs use asyncpg when it is installed, otherwise
psycopg 3 (async). An explicit postgresql+asyncpg:// or postgresql+psycopg://
URL is used as given. On Windows, psycopg async needs a SelectorEventLoop;
prefer asyncpg there.
"""
from __future__ import annotations

import importlib.util
import threading
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.ai.exceptions import ConfigurationError

_read_engine: AsyncEngine | None = None
_write_engine: AsyncEngine | None = None
_lock = threading.Lock()

APPLICATION_NAME = "vaakdb-ai"


def cfg(name: str, default: Any = None) -> Any:
    """Read a setting from app.ai.config.settings (imported lazily so the db
    modules can be imported, and unit-tested, without a full environment)."""
    from app.ai.config import settings

    value = getattr(settings, name, None)
    return default if value is None else value


def _default_driver() -> str:
    return "asyncpg" if importlib.util.find_spec("asyncpg") else "psycopg"


def _async_url(url: str) -> str:
    """Switch plain postgresql:// URLs (and sync drivers) to an async driver."""
    for prefix in ("postgresql+psycopg2://", "postgresql://", "postgres://"):
        if url.startswith(prefix):
            return f"postgresql+{_default_driver()}://" + url[len(prefix):]
    return url


def _connect_args(url: str, server_settings: dict[str, str]) -> dict[str, Any]:
    """Session-level settings, in the form each driver expects."""
    settings = {"application_name": APPLICATION_NAME, **server_settings}
    if url.startswith("postgresql+asyncpg://"):
        return {"server_settings": settings}
    options = " ".join(f"-c {k}={v}" for k, v in settings.items())
    return {"options": options}


def _build(url_setting: str, server_settings: dict[str, str], pool_size: int,
           max_overflow: int) -> AsyncEngine:
    url = cfg(url_setting)
    if not url:
        raise ConfigurationError(f"{url_setting} is not configured.")
    url = _async_url(str(url))
    return create_async_engine(
        url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        pool_recycle=int(cfg("DB_POOL_RECYCLE_S", 1800)),
        pool_timeout=10,
        connect_args=_connect_args(url, server_settings),
    )


def get_read_engine() -> AsyncEngine:
    global _read_engine
    if _read_engine is None:
        with _lock:
            if _read_engine is None:
                _read_engine = _build(
                    "DATABASE_READ_URL",
                    {
                        "default_transaction_read_only": "on",
                        "statement_timeout": str(int(cfg("SQL_TIMEOUT_MS", 5000))),
                        "idle_in_transaction_session_timeout": "15000",
                    },
                    pool_size=int(cfg("DB_POOL_SIZE", 5)),
                    max_overflow=int(cfg("DB_MAX_OVERFLOW", 5)),
                )
    return _read_engine


def get_write_engine() -> AsyncEngine:
    global _write_engine
    if _write_engine is None:
        with _lock:
            if _write_engine is None:
                _write_engine = _build(
                    "DATABASE_WRITE_URL",
                    {
                        "statement_timeout": str(int(cfg("WRITE_TIMEOUT_MS", 15000))),
                        "idle_in_transaction_session_timeout": "30000",
                    },
                    pool_size=max(2, int(cfg("DB_POOL_SIZE", 5)) // 2),
                    max_overflow=2,
                )
    return _write_engine


async def ping(engine: AsyncEngine) -> bool:
    """Health check: True if the database answers SELECT 1."""
    try:
        async with engine.connect() as conn:
            await conn.exec_driver_sql("SELECT 1")
        return True
    except Exception:
        return False


async def dispose_engines() -> None:
    """Call from the FastAPI shutdown/lifespan handler."""
    global _read_engine, _write_engine
    for engine in (_read_engine, _write_engine):
        if engine is not None:
            await engine.dispose()
    _read_engine = _write_engine = None
