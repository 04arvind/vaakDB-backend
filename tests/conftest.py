"""Test setup.

Database tests need a real PostgreSQL server:
- TEST_PG_SUPERUSER_URL=postgresql://postgres:pw@localhost:5432/postgres  (any server), or
- a running Docker daemon (testcontainers starts postgres:16).
Otherwise they are skipped and only the pure unit tests run.

For each run, fresh databases (meta + target) and fresh vaak_ro / vaak_rw style
roles are created from scripts/create_*_role.sql, and dropped afterwards.
Environment variables are set BEFORE any app module is imported.

Every user has a private workspace (schemas u<id>_<name>). Tests create a
user's tables through the write role, exactly like the app does, with
`ws_exec(caller, "CREATE TABLE {s}.students ...", schema="college")`.
"""
from __future__ import annotations

import os
import secrets
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(ROOT))

SUFFIX = secrets.token_hex(4)
META_DB = f"vaak_test_meta_{SUFFIX}"
TARGET_DB = f"vaak_test_target_{SUFFIX}"
RO_ROLE = f"vaak_test_ro_{SUFFIX}"
RW_ROLE = f"vaak_test_rw_{SUFFIX}"
RO_PW = secrets.token_urlsafe(16)
RW_PW = secrets.token_urlsafe(16)

_container: Any = None
_server_url: str | None = None  # libpq URL of a superuser connection (no database part)


def _libpq(url: str) -> str:
    from app.db.urls import normalize_pg_url

    return normalize_pg_url(url).replace("postgresql+psycopg://", "postgresql://", 1)


def _with_db(url: str, db: str) -> str:
    base = url.split("?")[0].rsplit("/", 1)[0]
    return f"{base}/{db}"


def _start_server() -> str | None:
    global _container
    url = os.getenv("TEST_PG_SUPERUSER_URL")
    if url:
        return _libpq(url)
    try:
        from testcontainers.postgres import PostgresContainer

        _container = PostgresContainer("postgres:16-alpine")
        _container.start()
        return _libpq(_container.get_connection_url())
    except Exception:
        return None


def _creds_url(url: str, user: str, password: str, db: str) -> str:
    from urllib.parse import quote

    host = url.split("@", 1)[1].split("/", 1)[0] if "@" in url else url.split("://", 1)[1].split("/")[0]
    return f"postgresql://{user}:{quote(password, safe='')}@{host}/{db}"


def pytest_configure(config: pytest.Config) -> None:
    global _server_url
    upload_dir = tempfile.mkdtemp(prefix="vaak_uploads_")
    os.environ.update({
        "APP_ENV": "test",
        "UPLOAD_DIR": upload_dir,
        "JWT_SECRET": "test-secret-" + "x" * 40,
        "AI_JWT_SECRET": "test-secret-" + "x" * 40,
        "MAX_INSERT_ROWS": "1000",
        "INSERT_BATCH_SIZE": "500",
        "OPERATION_TTL_MINUTES": "15",
        "DEFAULT_SCHEMA": "main",
        "MAX_SCHEMAS_PER_USER": "25",
        "ENABLE_AI_ROUTES": "false",
        "LLM_PROVIDER": "groq",
        "AI_AUTO_CREATE_TABLES": "false",
        "GROQ_API_KEY": "",  # tests never call a real LLM
        "GEMINI_API_KEY": "",
    })
    # never fall back to the developer's real databases
    for key in ("TARGET_DATABASE_URL", "DATABASE_URL", "DATABASE_READ_URL", "DATABASE_WRITE_URL"):
        os.environ[key] = ""

    _server_url = _start_server()
    if _server_url is None:
        os.environ["META_DATABASE_URL"] = "postgresql://nobody:none@127.0.0.1:1/none"
        os.environ["TARGET_DB_RO_URL"] = "postgresql://nobody:none@127.0.0.1:1/none"
        os.environ["TARGET_DB_RW_URL"] = "postgresql://nobody:none@127.0.0.1:1/none"
        return

    # Set every URL BEFORE importing anything that loads app settings, so the
    # developer's .env databases can never be touched by the test run.
    os.environ["META_DATABASE_URL"] = _with_db(_server_url, META_DB)
    os.environ["TARGET_DB_RO_URL"] = _creds_url(_server_url, RO_ROLE, RO_PW, TARGET_DB)
    os.environ["TARGET_DB_RW_URL"] = _creds_url(_server_url, RW_ROLE, RW_PW, TARGET_DB)

    import psycopg

    with psycopg.connect(_server_url, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{META_DB}"')
        conn.execute(f'CREATE DATABASE "{TARGET_DB}"')

    from app.core.config import settings
    from app.db.roles import apply_role_scripts

    assert META_DB in settings.database_url, "test settings must point at the test database"

    with psycopg.connect(_with_db(_server_url, TARGET_DB), autocommit=True) as conn:
        apply_role_scripts(conn, ro_role=RO_ROLE, rw_role=RW_ROLE, ro_password=RO_PW,
                           rw_password=RW_PW)
        # data outside every workspace, which nobody may reach through the API
        conn.execute("CREATE SCHEMA private")
        conn.execute("CREATE TABLE private.salaries (id int primary key, amount numeric)")
        conn.execute("CREATE TABLE public.users (id int primary key, password text)")

    from alembic.config import Config

    from alembic import command

    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")


def pytest_unconfigure(config: pytest.Config) -> None:
    if _server_url is None:
        return
    try:
        from app.db import session, target_db

        target_db.dispose_engines()
        session.engine.dispose()
    except Exception:
        pass
    import psycopg

    try:
        with psycopg.connect(_server_url, autocommit=True) as conn:
            for db in (META_DB, TARGET_DB):
                conn.execute(f'DROP DATABASE IF EXISTS "{db}" WITH (FORCE)')
            for role in (RO_ROLE, RW_ROLE):
                conn.execute(f'DROP ROLE IF EXISTS "{role}"')
    finally:
        if _container is not None:
            _container.stop()


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def pg_available() -> bool:
    return _server_url is not None


@pytest.fixture
def db_required(pg_available: bool) -> None:
    if not pg_available:
        pytest.skip("needs PostgreSQL (set TEST_PG_SUPERUSER_URL or start Docker)")


@pytest.fixture
def admin_target() -> Iterator[Any]:
    """Superuser connection to the target database (to inspect results)."""
    import psycopg

    assert _server_url is not None
    with psycopg.connect(_with_db(_server_url, TARGET_DB), autocommit=True) as conn:
        yield conn


@pytest.fixture
def admin_meta() -> Iterator[Any]:
    import psycopg

    assert _server_url is not None
    with psycopg.connect(_with_db(_server_url, META_DB), autocommit=True) as conn:
        yield conn


@pytest.fixture
def clean_target(db_required: None, admin_target: Any) -> Iterator[Any]:
    """Drops every workspace schema (u<id>_*) before and after a test."""

    def wipe() -> None:
        rows = admin_target.execute(
            "SELECT nspname FROM pg_namespace WHERE nspname ~ '^u[0-9]+_'").fetchall()
        for (name,) in rows:
            admin_target.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')

    wipe()
    yield admin_target
    wipe()


def make_user(email: str | None = None) -> Any:
    from app.db.models import User
    from app.db.session import SessionLocal
    from app.security.secrets import hash_password

    with SessionLocal() as db:
        user = User(email=email or f"user-{secrets.token_hex(4)}@example.com",
                    password_hash=hash_password("password123"))
        db.add(user)
        db.commit()
        db.refresh(user)
        return user


@pytest.fixture
def caller_for(db_required: None) -> Callable[[], Any]:
    """Creates a new user (with an empty workspace) and returns its Caller."""
    from app.security.authorization import Caller

    def make() -> Caller:
        return Caller(user_id=make_user().id)

    return make


def phys(caller: Any, schema: str) -> str:
    from app.security.authorization import to_physical

    return to_physical(caller, schema)


@pytest.fixture
def ws_exec(db_required: None) -> Callable[..., None]:
    """Runs SQL as the write role inside a user's workspace schema (created if
    needed): ws_exec(caller, "CREATE TABLE {s}.t (id int)", schema="college")."""
    import psycopg

    from app.core.config import settings

    url = str(settings.target_rw_url).replace("postgresql+psycopg://", "postgresql://", 1)

    def run(caller: Any, *statements: str, schema: str = "college") -> None:
        physical = phys(caller, schema)
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{physical}"')
            for sql in statements:
                conn.execute(sql.replace("{s}", f'"{physical}"'))

    return run
