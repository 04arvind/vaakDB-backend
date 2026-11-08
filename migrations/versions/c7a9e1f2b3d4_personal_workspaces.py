"""Personal workspaces: drop users.role (every user has full access to their own
workspace and none to anyone else's).

Revision ID: c7a9e1f2b3d4
Revises: b3f1c2d4e5a6
Create Date: 2026-11-08 10:00:00
"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa

from alembic import op

revision: str = "c7a9e1f2b3d4"
down_revision: Union[str, Sequence[str], None] = "b3f1c2d4e5a6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint("ck_users_role", "users", type_="check")
    op.drop_column("users", "role")


def downgrade() -> None:
    op.add_column(
        "users",
        sa.Column("role", sa.String(length=20), server_default="admin", nullable=False),
    )
    op.create_check_constraint(
        "ck_users_role", "users", "role IN ('viewer', 'analyst', 'editor', 'admin')"
    )
