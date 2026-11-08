"""Natural-language SELECT is confined to the caller's workspace, whatever SQL the model writes."""
from __future__ import annotations

from typing import Any

import pytest

from app.ai.agents.sql_agent import SQLAgent, SQLResult
from app.core.exceptions import AppError, ErrorCode
from app.security.authorization import Caller
from app.services.nl_query import run_workspace_query, scope_to_workspace


@pytest.fixture
def two_users(caller_for: Any, clean_target: Any, ws_exec: Any) -> tuple[Caller, Caller]:
    alice, bob = caller_for(), caller_for()
    ws_exec(alice, "CREATE TABLE {s}.students (id int PRIMARY KEY, name text)",
            "INSERT INTO {s}.students VALUES (1, 'alice-1'), (2, 'alice-2'), (3, 'alice-3')")
    ws_exec(bob, "CREATE TABLE {s}.students (id int PRIMARY KEY, name text)",
            "INSERT INTO {s}.students VALUES (1, 'bob-1')")
    return alice, bob


def fake_agent(monkeypatch: pytest.MonkeyPatch, sqls: list[str]) -> list[str]:
    """SQLAgent that returns the given SQL strings in order (generate, then repairs)."""
    seen: list[str] = []

    async def generate_sql(self: Any, question: str, schema: str, history: Any = None) -> SQLResult:
        seen.append(schema)
        return SQLResult(sql=sqls.pop(0))

    async def repair_sql(self: Any, question: str, schema: str, bad_sql: str, error: str) -> SQLResult:
        return SQLResult(sql=sqls.pop(0) if sqls else bad_sql)

    async def answer(self: Any, question: str, sql: str, rows: Any, max_rows: int = 50, *,
                     truncated: bool = False) -> str:
        return f"{len(rows)} row(s)"

    monkeypatch.setattr(SQLAgent, "generate_sql", generate_sql)
    monkeypatch.setattr(SQLAgent, "repair_sql", repair_sql)
    monkeypatch.setattr(SQLAgent, "answer", answer)
    return seen


async def test_reads_own_table(two_users: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    alice, _ = two_users
    seen = fake_agent(monkeypatch, ["SELECT name FROM students ORDER BY id"])
    result = await run_workspace_query(alice, "list students", "college")
    assert [r["name"] for r in result.rows] == ["alice-1", "alice-2", "alice-3"]
    assert alice.prefix not in result.sql and alice.prefix not in seen[0]  # logical names only


async def test_same_names_resolve_to_own_data(two_users: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _, bob = two_users
    fake_agent(monkeypatch, ["SELECT name FROM college.students"])
    result = await run_workspace_query(bob, "list students", "college")
    assert [r["name"] for r in result.rows] == ["bob-1"]


@pytest.mark.parametrize("attack", [
    "SELECT name FROM {alice}.students",
    "SELECT name FROM private.salaries",
    "SELECT * FROM public.users",
    "SELECT nspname FROM pg_catalog.pg_namespace",
    "SELECT name FROM students UNION SELECT name FROM {alice}.students",
    "WITH s AS (SELECT name FROM {alice}.students) SELECT * FROM s",
])
async def test_cannot_read_other_workspaces(two_users: Any, monkeypatch: pytest.MonkeyPatch,
                                            attack: str) -> None:
    alice, bob = two_users
    sql = attack.format(alice=f'"{alice.prefix}college"')
    fake_agent(monkeypatch, [sql, sql, sql])
    with pytest.raises(AppError) as exc:
        await run_workspace_query(bob, "show everything", "college")
    assert exc.value.code == ErrorCode.QUERY_FAILED


async def test_missing_schema(two_users: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    alice, _ = two_users
    fake_agent(monkeypatch, ["SELECT 1"])
    with pytest.raises(AppError) as exc:
        await run_workspace_query(alice, "q", "nothing_here")
    assert exc.value.code == ErrorCode.SCHEMA_NOT_FOUND


def test_scope_rewrites_every_table_reference() -> None:
    sql = ("WITH recent AS (SELECT id FROM students) "
           "SELECT s.name FROM college.students s JOIN recent r ON r.id = s.id")
    out = scope_to_workspace(sql, logical="college", physical="u9_college", allowed={"students"})
    assert out.count("u9_college.students") == 2 and "JOIN recent" in out  # CTE left alone
