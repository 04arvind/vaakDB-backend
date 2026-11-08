"""Create/refresh the vaak_ro and vaak_rw roles on target_db.

    python scripts/setup_roles.py --admin-url "postgresql://postgres:PASSWORD@localhost:5432/target_db"

User workspaces (schemas u<id>_<name>) are created by the app itself. Passwords
come from VAAK_RO_PASSWORD / VAAK_RW_PASSWORD or are generated; the resulting
TARGET_DB_RO_URL / TARGET_DB_RW_URL lines are printed once for your .env.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    import psycopg

    from app.core.config import settings
    from app.db.roles import apply_role_scripts
    from app.db.urls import _split, normalize_pg_url
    from app.security.secrets import generate_secret

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--admin-url", default=os.getenv("TARGET_ADMIN_URL") or settings.target_database_url,
                        help="superuser URL for target_db (default: TARGET_ADMIN_URL / TARGET_DATABASE_URL)")
    parser.add_argument("--ro-role", default="vaak_ro")
    parser.add_argument("--rw-role", default="vaak_rw")
    args = parser.parse_args()

    if not args.admin_url:
        parser.error("--admin-url is required")

    ro_pw = os.getenv("VAAK_RO_PASSWORD") or generate_secret(24)
    rw_pw = os.getenv("VAAK_RW_PASSWORD") or generate_secret(24)
    libpq_url = normalize_pg_url(args.admin_url).replace("postgresql+psycopg://", "postgresql://", 1)

    with psycopg.connect(libpq_url, autocommit=True) as conn:
        apply_role_scripts(conn, ro_role=args.ro_role, rw_role=args.rw_role, ro_password=ro_pw,
                           rw_password=rw_pw)
        db = conn.execute("SELECT current_database()").fetchone()[0]  # type: ignore[index]

    _, _, host, _ = _split(args.admin_url)
    print(f"Roles {args.ro_role} / {args.rw_role} ready on {db}")
    print("Add to .env (shown once):")
    print(f"TARGET_DB_RO_URL=postgresql://{args.ro_role}:{quote(ro_pw, safe='')}@{host}/{db}")
    print(f"TARGET_DB_RW_URL=postgresql://{args.rw_role}:{quote(rw_pw, safe='')}@{host}/{db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
