"""meta_db (application database) helpers.

The engine and session factory live in app.db.session; this module adds the
small helpers the API needs (health check, a session that commits on its own
for audit records)."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import session as _session


@contextmanager
def independent_session() -> Iterator[Session]:
    """A separate session that commits on exit. Used for audit rows, which must
    be written even when the caller's transaction rolls back."""
    db = _session.SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def ping() -> bool:
    try:
        with _session.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
