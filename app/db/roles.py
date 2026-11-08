"""Applies scripts/create_ro_role.sql + scripts/create_rw_role.sql (used by
scripts/setup_roles.py and the test suite)."""
from __future__ import annotations

import re
from pathlib import Path

from app.core.config import PROJECT_ROOT

SCRIPTS = PROJECT_ROOT / "scripts"
_ROLE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def render_script(path: Path, *, ro_role: str, rw_role: str, ro_password: str,
                  rw_password: str) -> str:
    for role in (ro_role, rw_role):
        if not _ROLE.match(role):
            raise ValueError(f"invalid role name {role!r}")

    def lit(value: str) -> str:  # value goes inside a '...' SQL literal
        if "\x00" in value:
            raise ValueError("NUL in value")
        return value.replace("'", "''")

    text = path.read_text(encoding="utf-8")
    values = {"RO_ROLE": ro_role, "RW_ROLE": rw_role, "RO_PASSWORD": lit(ro_password),
              "RW_PASSWORD": lit(rw_password)}
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def apply_role_scripts(conn: object, *, ro_role: str, rw_role: str, ro_password: str,
                       rw_password: str) -> None:
    """conn: an autocommit psycopg connection as a superuser, on target_db."""
    for name in ("create_ro_role.sql", "create_rw_role.sql"):
        sql = render_script(SCRIPTS / name, ro_role=ro_role, rw_role=rw_role,
                            ro_password=ro_password, rw_password=rw_password)
        conn.execute(sql)  # type: ignore[attr-defined]
