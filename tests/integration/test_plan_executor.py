"""Plan executor against real PostgreSQL, as the restricted vaak_rw-style role,
inside per-user workspaces."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from app.core.exceptions import AppError, ErrorCode
from app.schemas.plans import CreateTablePlan, DeleteRowsPlan, InsertRowsPlan, UpdateRowsPlan
from app.security.authorization import Caller
from app.services.plan_executor import execute_plan
from app.services.plan_validator import ValidatedPlan, validate_plan
from tests.conftest import phys

pytestmark = pytest.mark.usefixtures("clean_target")
ROOT = Path(__file__).resolve().parents[2]

COLS = [{"name": "roll_no", "pg_type": "integer", "primary_key": True},
        {"name": "name", "pg_type": "varchar(50)"},
        {"name": "dob", "pg_type": "date"},
        {"name": "gpa", "pg_type": "numeric(4,2)"}]


def rows(n: int) -> list[list[str]]:
    return [[str(i), f"student {i}", "2004-05-14", "8.5"] for i in range(1, n + 1)]


def validated(plan: Any, caller: Caller) -> ValidatedPlan:
    from app.db.target_db import get_read_engine

    with get_read_engine().connect() as conn:
        return validate_plan(plan, caller=caller, conn=conn)


def create(n: int = 5, **kw: Any) -> CreateTablePlan:
    return CreateTablePlan.model_validate({"schema": "college", "table": "students",
                                           "columns": COLS, "rows": rows(n), **kw})


def regclass(conn: Any, caller: Caller, table: str = "students", schema: str = "college") -> Any:
    return conn.execute("SELECT to_regclass(%s)", (f'"{phys(caller, schema)}".{table}',)).fetchone()[0]


def schema_exists(conn: Any, caller: Caller, schema: str = "college") -> bool:
    return conn.execute("SELECT count(*) FROM pg_namespace WHERE nspname = %s",
                        (phys(caller, schema),)).fetchone()[0] == 1


@pytest.fixture
def alice(caller_for: Any) -> Caller:
    return caller_for()


def test_create_schema_and_table(clean_target: Any, alice: Caller) -> None:
    result = execute_plan(validated(create(5), alice))
    assert result["schema_created"] and result["table_created"] and result["rows_written"] == 5
    assert result["target"] == {"schema": "college", "table": "students"}
    table = f'"{phys(alice, "college")}".students'
    got = clean_target.execute(f"SELECT roll_no, name, dob::text, gpa FROM {table} ORDER BY roll_no"
                               ).fetchall()
    assert got[0] == (1, "student 1", "2004-05-14", Decimal("8.50")) and len(got) == 5
    owner = clean_target.execute("SELECT tableowner FROM pg_tables WHERE schemaname = %s",
                                 (phys(alice, "college"),)).fetchone()[0]
    assert owner.startswith("vaak_test_rw_")  # written by the restricted role, not a superuser


def test_read_role_sees_new_workspace_tables(clean_target: Any, alice: Caller) -> None:
    from sqlalchemy import text

    from app.db.target_db import get_read_engine

    execute_plan(validated(create(2), alice))
    with get_read_engine().connect() as conn:  # default privileges: vaak_ro can read it
        n = conn.execute(text(f'SELECT count(*) FROM "{phys(alice, "college")}".students')).scalar()
    assert n == 2


def test_batches_of_configured_size(alice: Caller, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "insert_batch_size", 2)
    batches: list[tuple[int, int]] = []
    execute_plan(validated(create(5), alice), after_batch=lambda n, total: batches.append((n, total)))
    assert batches == [(1, 2), (2, 4), (3, 5)]


def test_rollback_on_mid_insert_failure(clean_target: Any, alice: Caller,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "insert_batch_size", 2)

    def boom(batch: int, total: int) -> None:
        if batch == 2:
            raise RuntimeError("simulated crash after 4 rows")

    with pytest.raises(RuntimeError):
        execute_plan(validated(create(6), alice), after_batch=boom)
    # CREATE SCHEMA + CREATE TABLE rolled back together with the rows
    assert regclass(clean_target, alice) is None and not schema_exists(clean_target, alice)


def test_rollback_on_database_error(clean_target: Any, alice: Caller, ws_exec: Any) -> None:
    ws_exec(alice, "CREATE TABLE {s}.students (roll_no int PRIMARY KEY, name text)",
            "INSERT INTO {s}.students VALUES (3, 'existing')")
    plan = InsertRowsPlan.model_validate({"schema": "college", "table": "students",
                                          "columns": ["roll_no", "name"],
                                          "rows": [["1", "a"], ["2", "b"], ["3", "dup"]]})
    with pytest.raises(AppError) as exc:
        execute_plan(validated(plan, alice))
    assert "duplicate" in exc.value.message and "rolled back" in exc.value.message
    assert "INSERT" not in exc.value.message  # no SQL leaked
    n = clean_target.execute(f'SELECT count(*) FROM "{phys(alice, "college")}".students').fetchone()[0]
    assert n == 1


def test_killed_process_leaves_nothing(clean_target: Any, alice: Caller) -> None:
    """Hard-kill the process in the middle of a multi-batch insert."""
    script = textwrap.dedent(
        """
        import sys, time
        sys.path.insert(0, %r)
        from app.core.config import settings
        settings.insert_batch_size = 2
        from app.schemas.plans import CreateTablePlan
        from app.db.target_db import get_read_engine
        from app.security.authorization import Caller
        from app.services.plan_validator import validate_plan
        from app.services.plan_executor import execute_plan
        rows = [[str(i), "s", "2004-01-01", "1.0"] for i in range(1, 11)]
        plan = CreateTablePlan.model_validate({"schema": "college", "table": "killed",
            "columns": %r, "rows": rows})
        with get_read_engine().connect() as conn:
            v = validate_plan(plan, caller=Caller(user_id=%d), conn=conn)
        def pause(batch, total):
            if batch == 2:
                print("MID-INSERT", flush=True)
                time.sleep(60)
        execute_plan(v, after_batch=pause)
        """
    ) % (str(ROOT), COLS, alice.user_id)
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=dict(os.environ))
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        assert "MID-INSERT" in line, proc.stderr.read() if proc.stderr else ""
        assert regclass(clean_target, alice, "killed") is None  # uncommitted: invisible
        proc.kill()  # SIGKILL / TerminateProcess: no cleanup code runs
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    for _ in range(50):  # the server notices the dropped connection and aborts the transaction
        busy = clean_target.execute(
            "SELECT count(*) FROM pg_stat_activity WHERE application_name = 'vaakdb-backend-rw' "
            "AND state LIKE 'idle in transaction%%'").fetchone()[0]
        if not busy:
            break
        time.sleep(0.2)
    assert regclass(clean_target, alice, "killed") is None
    assert not schema_exists(clean_target, alice)


def test_append_update_delete_replace(clean_target: Any, alice: Caller) -> None:
    execute_plan(validated(create(3), alice))
    append = create(0, if_exists="append").model_copy(update={"rows": [["10", "new", None, None]]})
    appended = execute_plan(validated(append, alice))
    assert appended["table_created"] is False and appended["rows_written"] == 1
    assert "schema_created" not in appended

    upd = UpdateRowsPlan.model_validate({"schema": "college", "table": "students",
                                         "set": [{"column": "gpa", "value": "9.99"}],
                                         "where": [{"column": "roll_no", "op": "<=", "value": 2}]})
    assert execute_plan(validated(upd, alice))["rows_updated"] == 2

    dele = DeleteRowsPlan.model_validate({"schema": "college", "table": "students",
                                          "where": [{"column": "gpa", "op": "=", "value": "9.99"}]})
    out = execute_plan(validated(dele, alice))
    assert out["rows_deleted"] == 2 and out["preview_affected_rows"] == 2

    out = execute_plan(validated(create(1, if_exists="replace"), alice))
    assert out["table_replaced"] and out["rows_written"] == 1
    n = clean_target.execute(f'SELECT count(*) FROM "{phys(alice, "college")}".students').fetchone()[0]
    assert n == 1


def test_users_tables_are_independent(clean_target: Any, caller_for: Any) -> None:
    alice, bob = caller_for(), caller_for()
    execute_plan(validated(create(3), alice))
    execute_plan(validated(create(1), bob))
    count = "SELECT count(*) FROM \"{}\".students"
    assert clean_target.execute(count.format(phys(alice, "college"))).fetchone()[0] == 3
    assert clean_target.execute(count.format(phys(bob, "college"))).fetchone()[0] == 1


def test_table_created_between_preview_and_execute(alice: Caller, ws_exec: Any) -> None:
    v = validated(create(2), alice)
    ws_exec(alice, "CREATE TABLE {s}.students (x int)")
    with pytest.raises(AppError) as exc:
        execute_plan(v)
    assert exc.value.code == ErrorCode.TABLE_EXISTS


def test_write_role_cannot_leave_workspaces(clean_target: Any, alice: Caller) -> None:
    """Even a forged plan that skipped validation cannot touch public/private:
    the database role itself has no privileges there."""
    execute_plan(validated(create(1), alice))
    v = validated(DeleteRowsPlan.model_validate({
        "schema": "college", "table": "students",
        "where": [{"column": "roll_no", "op": "=", "value": 1}]}), alice)
    clean_target.execute("INSERT INTO public.users VALUES (1, 'x') ON CONFLICT DO NOTHING")
    for schema, table in (("public", "users"), ("private", "salaries")):
        forged = ValidatedPlan(
            plan=v.plan.model_copy(update={"schema_name": schema, "table": table}),
            table_exists=True, columns=[], existing=v.existing, column_types=v.column_types,
        )
        forged.plan.where[0].column = "id"  # type: ignore[union-attr]
        forged.column_types = {"id": v.column_types["roll_no"]}
        with pytest.raises(AppError) as exc:
            execute_plan(forged)
        assert exc.value.code == ErrorCode.PERMISSION_DENIED
    assert clean_target.execute("SELECT count(*) FROM public.users").fetchone()[0] == 1
    clean_target.execute("DELETE FROM public.users")


def test_rw_role_cannot_escape(clean_target: Any) -> None:
    import psycopg

    from app.core.config import settings

    url = str(settings.target_rw_url).replace("postgresql+psycopg://", "postgresql://")
    clean_target.execute("CREATE SCHEMA IF NOT EXISTS admin_area")
    clean_target.execute("CREATE TABLE IF NOT EXISTS admin_area.t (a int)")
    with psycopg.connect(url, autocommit=True) as rw:
        for sql in ("CREATE TABLE public.x (a int)", "SELECT * FROM private.salaries",
                    "TRUNCATE admin_area.t", "DROP TABLE admin_area.t",
                    "ALTER TABLE admin_area.t ADD b int", "CREATE TABLE admin_area.y (a int)"):
            with pytest.raises((psycopg.errors.InsufficientPrivilege, psycopg.errors.UndefinedTable)):
                rw.execute(sql)
        assert rw.execute("SELECT rolsuper FROM pg_roles WHERE rolname = current_user"
                          ).fetchone()[0] is False
    clean_target.execute("DROP SCHEMA admin_area CASCADE")
