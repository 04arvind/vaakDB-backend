"""DDL for the memory tables. Run ensure_memory_schema() once at startup (or copy
the statements into an Alembic migration).

All table names start with "ai_", so schema_inspector hides them from the SQL
agent. Do NOT grant the read-only Postgres role access to them: chat history and
memories belong to individual users and must never be reachable by LLM-written SQL.
Grant SELECT/INSERT/UPDATE/DELETE on these tables to the WRITE role only.
"""
from __future__ import annotations

from app.ai.db.engine import get_write_engine

STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS ai_conversations (
         id         TEXT PRIMARY KEY,
         user_id    TEXT NOT NULL,
         title      TEXT,
         created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
         updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
       )""",
    "CREATE INDEX IF NOT EXISTS ai_conversations_user_idx "
    "ON ai_conversations (user_id, updated_at DESC)",
    """CREATE TABLE IF NOT EXISTS ai_messages (
         id              BIGSERIAL PRIMARY KEY,
         conversation_id TEXT NOT NULL REFERENCES ai_conversations(id) ON DELETE CASCADE,
         role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
         content         TEXT NOT NULL,
         meta            JSONB NOT NULL DEFAULT '{}'::jsonb,
         created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
       )""",
    "CREATE INDEX IF NOT EXISTS ai_messages_conv_idx ON ai_messages (conversation_id, id)",
    """CREATE TABLE IF NOT EXISTS ai_memories (
         id             BIGSERIAL PRIMARY KEY,
         user_id        TEXT NOT NULL,
         content        TEXT NOT NULL,
         category       TEXT NOT NULL DEFAULT 'other',
         importance     SMALLINT NOT NULL DEFAULT 3,
         source_session TEXT,
         created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
         updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
         UNIQUE (user_id, content)
       )""",
    "CREATE INDEX IF NOT EXISTS ai_memories_user_idx "
    "ON ai_memories (user_id, importance DESC, updated_at DESC)",
]


async def ensure_memory_schema() -> None:
    async with get_write_engine().begin() as conn:
        for statement in STATEMENTS:  # one statement per call (asyncpg limitation)
            await conn.exec_driver_sql(statement)