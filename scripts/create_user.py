"""Create a user (or reset an existing user's password) in meta_db.
Every user gets their own private workspace.

    python scripts/create_user.py you@example.com
The password is read from VAAK_USER_PASSWORD or prompted for.
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    from sqlalchemy import func, select

    from app.db.models import User
    from app.db.session import SessionLocal
    from app.security.secrets import MIN_PASSWORD_LENGTH, hash_password

    parser = argparse.ArgumentParser()
    parser.add_argument("email")
    args = parser.parse_args()

    password = os.getenv("VAAK_USER_PASSWORD") or getpass.getpass("Password: ")
    if len(password) < MIN_PASSWORD_LENGTH:
        parser.error(f"password must be at least {MIN_PASSWORD_LENGTH} characters")

    with SessionLocal() as db:
        stmt = select(User).where(func.lower(User.email) == args.email.lower())
        user = db.execute(stmt).scalar_one_or_none()
        if user is None:
            user = User(email=args.email.lower(), password_hash=hash_password(password))
            db.add(user)
            action = "created"
        else:
            user.password_hash = hash_password(password)
            action = "updated"
        db.commit()
        print(f"User {user.email} {action} (id={user.id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
