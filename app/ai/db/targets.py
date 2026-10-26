"""Write targets for /ai/ingest: the ONLY tables the AI module may insert into.

register_targets() runs once at startup (see app.ai.startup). Add one
register_target() call per table you want users to fill from text or files.
Nothing is registered by default, so ingest returns "Unknown ingest target"
until you opt in.

Example, for a table
    CREATE TABLE expenses (id BIGSERIAL PRIMARY KEY, user_id TEXT NOT NULL,
                           vendor TEXT, amount NUMERIC(12,2), spent_on DATE,
                           UNIQUE (user_id, vendor, amount, spent_on));

    class Expense(BaseModel):
        vendor: str | None = Field(default=None, max_length=200)
        amount: float | None = Field(default=None, ge=0)
        spent_on: date | None = None

    register_target(
        "expenses", table="expenses", model=Expense,
        owner_column="user_id",
        conflict_columns=("user_id", "vendor", "amount", "spent_on"),
        hints="Amounts are in INR.",
    )

Grant the write role INSERT (and UPDATE if updatable_columns is set) on each
table, plus USAGE on its id sequence. Nothing else.
"""
from __future__ import annotations

_registered = False


def register_targets() -> None:
    """Idempotent: safe to call from several startup paths."""
    global _registered
    if _registered:
        return
    # register_target("expenses", table="expenses", model=Expense, owner_column="user_id")
    _registered = True
