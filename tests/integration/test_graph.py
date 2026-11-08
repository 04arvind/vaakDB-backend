"""Chat graph with a fake LLM: routing for each intent, repair loop, clarify path,
all inside the caller's own workspace."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.ai.dbops.llm import LLMUnavailableError
from app.security.authorization import Caller
from app.services.chat_service import run_chat
from app.services.nl_query import WorkspaceQueryResult
from tests.conftest import phys
from tests.fakes import STUDENTS_DRAFT, FakeChatModel, deps_with

FIX = Path(__file__).resolve().parents[1] / "fixtures"


def route(intent: str, schema: str | None = "college", table: str | None = "students",
          **extra: Any) -> dict[str, Any]:
    return {"intent": intent, "schema": schema, "table": table, **extra}


@pytest.fixture
def user(caller_for: Any, clean_target: Any) -> Caller:
    return caller_for()


@pytest.fixture
def students(user: Caller, ws_exec: Any) -> Caller:
    ws_exec(user, "CREATE TABLE {s}.students (roll_no int PRIMARY KEY, name text NOT NULL, age int)",
            "INSERT INTO {s}.students VALUES (1,'Asha',20),(2,'Rahul',21),(3,'Meera',19)")
    return user


def upload_students(caller: Any, name: str = "students.pdf") -> str:
    from app.db.session import SessionLocal
    from app.services.upload_service import handle_upload

    with SessionLocal() as db, open(FIX / name, "rb") as fh:
        return handle_upload(db, caller, fh, filename=name, content_type="application/pdf")["upload_id"]


async def chat(caller: Any, fake: FakeChatModel, message: str, **kw: Any) -> dict[str, Any]:
    run_select = kw.pop("run_select", None)
    return await run_chat(caller, message, deps=deps_with(fake, run_select=run_select), **kw)


# ---- metadata intents (no LLM for execution) ----------------------------- #
async def test_list_schemas_by_rule(students: Caller) -> None:
    fake = FakeChatModel()
    env = await chat(students, fake, "show all schemas")
    assert env["type"] == "schema_list" and fake.calls == []
    assert [i["schema"] for i in env["data"]["items"]] == ["college"]


async def test_list_tables_via_llm_router(students: Caller) -> None:
    fake = FakeChatModel({"IntentDecision": [route("list_tables", table=None)]})
    env = await chat(students, fake, "what do we store for the college dept?")
    assert env["type"] == "table_list" and env["data"]["items"][0]["table"] == "students"
    assert "college" in fake.calls_for("IntentDecision")[0].system  # router sees own schemas


async def test_describe_by_rule(students: Caller) -> None:
    env = await chat(students, FakeChatModel(), "What columns does college.students have?")
    assert env["type"] == "table_description"
    assert [c["name"] for c in env["data"]["columns"]] == ["roll_no", "name", "age"]


async def test_single_schema_is_used_when_none_named(students: Caller) -> None:
    fake = FakeChatModel({"IntentDecision": [route("list_tables", schema=None, table=None)]})
    env = await chat(students, fake, "list the tables")
    assert env["type"] == "table_list" and env["data"]["schema"] == "college"


async def test_empty_workspace_message(user: Caller) -> None:
    fake = FakeChatModel({"IntentDecision": [route("list_tables", schema=None, table=None)]})
    env = await chat(user, fake, "list the tables")
    assert env["type"] == "message" and "empty" in env["message"]


# ---- select ---------------------------------------------------------------- #
async def test_select_runs_in_own_workspace(students: Caller) -> None:
    fake = FakeChatModel({"IntentDecision": [route("select_query", table=None)]})
    calls: list[tuple[int, str]] = []

    async def run_select(caller: Caller, message: str, schema: str) -> WorkspaceQueryResult:
        calls.append((caller.user_id, schema))
        return WorkspaceQueryResult(sql="SELECT count(*) FROM students LIMIT 100",
                                    answer="There are 3 students.", columns=["count"],
                                    rows=[{"count": 3}])

    env = await chat(students, fake, "how many students are there?", run_select=run_select)
    assert env["type"] == "query_result" and env["data"]["rows"] == [{"count": 3}]
    assert env["message"] == "There are 3 students."
    assert calls == [(students.user_id, "college")]


# ---- write intents --------------------------------------------------------- #
async def test_ingest_file_preview(user: Caller, clean_target: Any) -> None:
    upload_id = upload_students(user)
    fake = FakeChatModel({"IntentDecision": [route("ingest_file")], "IngestDraft": [STUDENTS_DRAFT]})
    env = await chat(user, fake, "store this as students in schema college", upload_id=upload_id)
    assert env["type"] == "operation_preview", env
    data = env["data"]
    assert data["target"] == {"schema": "college", "table": "students"}
    assert [c["name"] for c in data["columns"]] == ["roll_no", "name", "age", "date_of_birth", "gpa"]
    assert data["total_rows"] == 5 and data["rows_to_write"] == 5 and data["failed_rows"] == []
    assert data["sample_rows"][0] == {"roll_no": 101, "name": "Asha Verma", "age": 20,
                                      "date_of_birth": "2004-05-14", "gpa": "8.70"}
    assert data["ddl"][0] == "CREATE SCHEMA college"
    assert not any(user.prefix in s for s in data["ddl"])
    assert "sample_rows" in fake.calls_for("IngestDraft")[0].user
    exists = clean_target.execute("SELECT count(*) FROM pg_namespace WHERE nspname = %s",
                                  (phys(user, "college"),)).fetchone()[0]
    assert exists == 0  # nothing touched the database before confirm


async def test_write_without_schema_uses_default(user: Caller) -> None:
    create = {"schema": "main", "table": "courses",
              "columns": [{"name": "id", "pg_type": "integer", "primary_key": True}]}
    fake = FakeChatModel({"IntentDecision": [route("create_table", schema=None, table="courses")],
                          "CreateTablePlan": [create]})
    env = await chat(user, fake, "create a courses table with an id")
    assert env["type"] == "operation_preview"
    assert env["data"]["target"] == {"schema": "main", "table": "courses"}


async def test_attached_file_wins_over_create_intent(user: Caller) -> None:
    upload_id = upload_students(user)
    fake = FakeChatModel({"IntentDecision": [route("create_table")], "IngestDraft": [STUDENTS_DRAFT]})
    env = await chat(user, fake, "make a students table in college from this", upload_id=upload_id)
    assert env["type"] == "operation_preview" and fake.calls_for("CreateTablePlan") == []


async def test_all_write_operations_for_every_user(students: Caller, ws_exec: Any) -> None:
    create = {"schema": "college", "table": "courses",
              "columns": [{"name": "id", "pg_type": "integer", "primary_key": True},
                          {"name": "title", "pg_type": "text"}]}
    fake = FakeChatModel({"IntentDecision": [route("create_table", table="courses")],
                          "CreateTablePlan": [create]})
    env = await chat(students, fake, "create a courses table in college with id and title")
    assert env["type"] == "operation_preview" and env["data"]["operation_type"] == "create_table"

    fake = FakeChatModel({"IntentDecision": [route("insert_rows")], "InsertRowsPlan": [
        {"schema": "college", "table": "students", "columns": ["roll_no", "name", "age"],
         "rows": [[4, "Zoya", 22]]}]})
    env = await chat(students, fake, "add Zoya, 22, roll 4 to college.students")
    assert env["data"]["rows_to_write"] == 1
    assert "students" in fake.calls_for("InsertRowsPlan")[0].user  # own table description given

    fake = FakeChatModel({"IntentDecision": [route("update_rows")], "UpdateRowsPlan": [
        {"schema": "college", "table": "students", "set": [{"column": "age", "value": 21}],
         "where": [{"column": "age", "op": "<", "value": 21}]}]})
    env = await chat(students, fake, "set age to 21 for students younger than 21 in college.students")
    assert env["data"]["affected_rows"] == 2

    fake = FakeChatModel({"IntentDecision": [route("delete_rows")], "DeleteRowsPlan": [
        {"schema": "college", "table": "students",
         "where": [{"column": "roll_no", "op": "=", "value": 3}]}]})
    env = await chat(students, fake, "delete roll 3 from college.students")
    assert env["type"] == "operation_preview" and env["data"]["affected_rows"] == 1


async def test_cannot_write_into_another_users_table(students: Caller, caller_for: Any) -> None:
    bob = caller_for()
    for schema in ("college", f"{students.prefix}college"):
        fake = FakeChatModel({"IntentDecision": [route("delete_rows", schema=schema)],
                              "DeleteRowsPlan": [{"schema": schema, "table": "students",
                                                  "where": [{"column": "roll_no", "op": "=",
                                                             "value": 1}]}]})
        env = await chat(bob, fake, f"delete roll 1 from {schema}.students")
        assert env["type"] == "error" and env["data"]["code"] == "TABLE_NOT_FOUND"


# ---- clarify / unsupported / errors --------------------------------------- #
async def test_clarify_when_table_missing(user: Caller) -> None:
    fake = FakeChatModel({"IntentDecision": [route("ingest_file", schema=None, table=None)]})
    env = await chat(user, fake, "save this", upload_id=upload_students(user))
    assert env["type"] == "message" and env["data"]["clarification"]
    assert env["message"].count("?") == 1 and fake.calls_for("IngestDraft") == []


async def test_clarify_from_router(user: Caller) -> None:
    fake = FakeChatModel({"IntentDecision": [route(
        "clarify", clarification_question="Which table should I update?")]})
    env = await chat(user, fake, "update it")
    assert env["message"] == "Which table should I update?"


async def test_ingest_without_upload_asks_for_file(user: Caller) -> None:
    fake = FakeChatModel({"IntentDecision": [route("ingest_file")]})
    env = await chat(user, fake, "import the attached file into college.students")
    assert env["type"] == "message" and "attach" in env["message"]


async def test_unsupported(user: Caller) -> None:
    fake = FakeChatModel({"IntentDecision": [route("unsupported")]})
    env = await chat(user, fake, "drop table college.students")
    assert env["type"] == "message" and "not available" in env["message"]


async def test_llm_unavailable(user: Caller) -> None:
    fake = FakeChatModel({"IntentDecision": [LLMUnavailableError("429")]})
    env = await chat(user, fake, "please do something with my data")
    assert env["data"]["code"] == "LLM_UNAVAILABLE" and "429" not in env["message"]


# ---- repair loop ----------------------------------------------------------- #
BAD_DRAFT = {**STUDENTS_DRAFT, "columns": [
    {**c, "pg_type": "integer"} if c["name"] == "name" else c for c in STUDENTS_DRAFT["columns"]]}


async def test_repair_loop_fixes_plan(user: Caller) -> None:
    fake = FakeChatModel({"IntentDecision": [route("ingest_file")],
                          "IngestDraft": [BAD_DRAFT, STUDENTS_DRAFT]}, consume_last=True)
    env = await chat(user, fake, "store this as students in college", upload_id=upload_students(user))
    assert env["type"] == "operation_preview"
    calls = fake.calls_for("IngestDraft")
    assert len(calls) == 2 and "previous plan was rejected" in calls[1].system


async def test_repair_loop_gives_up_after_two_attempts(user: Caller) -> None:
    fake = FakeChatModel({"IntentDecision": [route("ingest_file")],
                          "IngestDraft": [BAD_DRAFT, BAD_DRAFT, STUDENTS_DRAFT]}, consume_last=True)
    env = await chat(user, fake, "store this as students in college", upload_id=upload_students(user))
    assert env["type"] == "error" and env["data"]["code"] == "TYPE_CAST_FAILED"
    assert len(fake.calls_for("IngestDraft")) == 2


async def test_non_repairable_error_is_not_retried(students: Caller) -> None:
    fake = FakeChatModel({"IntentDecision": [route("ingest_file")], "IngestDraft": [STUDENTS_DRAFT]})
    env = await chat(students, fake, "store this as students in college",
                     upload_id=upload_students(students))
    assert env["data"]["code"] == "TABLE_EXISTS" and len(fake.calls_for("IngestDraft")) == 1
    assert students.prefix not in env["message"]
