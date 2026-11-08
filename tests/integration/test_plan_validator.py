"""Plan validator against a real PostgreSQL, inside per-user workspaces (>= 25 cases)."""
from __future__ import annotations

from typing import Any

import pytest

from app.core.exceptions import AppError, ErrorCode
from app.schemas.plans import CreateTablePlan, DeleteRowsPlan, InsertRowsPlan, UpdateRowsPlan
from app.security.authorization import Caller
from app.services.plan_validator import ValidatedPlan, validate_plan

pytestmark = pytest.mark.usefixtures("clean_target")

COLS = [
    {"name": "roll_no", "pg_type": "integer", "primary_key": True},
    {"name": "name", "pg_type": "varchar(50)"},
    {"name": "dob", "pg_type": "date"},
]
ROWS = [["1", "Asha", "2004-05-14"], ["2", "Rahul", "2003-11-02"]]


def create(**kw: Any) -> CreateTablePlan:
    data = {"schema": "college", "table": "students", "columns": COLS, "rows": ROWS, **kw}
    return CreateTablePlan.model_validate(data)


def run(plan: Any, caller: Caller) -> ValidatedPlan:
    from app.db.target_db import get_read_engine

    with get_read_engine().connect() as conn:
        return validate_plan(plan, caller=caller, conn=conn)


def code_of(plan: Any, caller: Caller) -> ErrorCode:
    with pytest.raises(AppError) as exc:
        run(plan, caller)
    return exc.value.code


@pytest.fixture
def alice(caller_for: Any) -> Caller:
    return caller_for()


@pytest.fixture
def bob(caller_for: Any) -> Caller:
    return caller_for()


@pytest.fixture
def students(alice: Caller, ws_exec: Any) -> Caller:
    """alice owns college.students with 3 rows."""
    ws_exec(alice,
            "CREATE TABLE {s}.students (roll_no int PRIMARY KEY, name varchar(50) NOT NULL, "
            "age int, dob date, notes jsonb)",
            "INSERT INTO {s}.students VALUES (1,'Asha',20,'2004-05-14',null),"
            "(2,'Rahul',21,'2003-11-02',null),(3,'Meera',19,null,null)")
    return alice


# ---- create ---------------------------------------------------------------- #
def test_01_create_in_new_schema(alice: Caller) -> None:
    v = run(create(), alice)
    assert v.rows_to_write == 2 and not v.table_exists and v.create_schema
    assert v.statements[0] == "CREATE SCHEMA college"
    assert any("CREATE TABLE college.students" in s for s in v.statements)
    assert not any(alice.prefix in s for s in v.statements)  # users only see their own names
    p = v.preview()
    assert p["target"] == {"schema": "college", "table": "students"}
    assert p["sample_rows"][0] == {"roll_no": 1, "name": "Asha", "dob": "2004-05-14"}
    assert v.plan.schema_name == f"{alice.prefix}college"


def test_02_system_schemas_rejected(alice: Caller) -> None:
    assert code_of(create(schema="information_schema"), alice) == ErrorCode.FORBIDDEN_SCHEMA


def test_03_schema_name_too_long(alice: Caller) -> None:
    assert code_of(create(schema="s" * 62), alice) == ErrorCode.INVALID_IDENTIFIER


def test_04_schema_quota(students: Caller, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "max_schemas_per_user", 1)
    assert code_of(create(schema="other"), students) == ErrorCode.PLAN_INVALID
    assert run(create(table="more"), students).create_schema is False  # existing schema is fine


def test_05_users_have_separate_namespaces(students: Caller, bob: Caller) -> None:
    """bob can create college.students too: it is a different table in his workspace."""
    v = run(create(), bob)
    assert not v.table_exists and v.plan.schema_name == f"{bob.prefix}college"


def test_06_table_exists_fail(students: Caller) -> None:
    assert code_of(create(), students) == ErrorCode.TABLE_EXISTS


def test_07_append_casts_to_existing_types(students: Caller) -> None:
    plan = create(if_exists="append", columns=[{"name": "roll_no", "pg_type": "text"},
                                               {"name": "name", "pg_type": "text"}],
                  rows=[["10", "Zed"]])
    v = run(plan, students)
    assert v.table_exists and v.cast.rows == [[10, "Zed"]]
    assert any("cast to the existing type" in w for w in v.warnings)


def test_08_append_unknown_column(students: Caller) -> None:
    plan = create(if_exists="append", columns=[{"name": "roll_no", "pg_type": "integer"},
                                               {"name": "name", "pg_type": "text"},
                                               {"name": "shoe_size", "pg_type": "integer"}],
                  rows=[["9", "X", "42"]])
    with pytest.raises(AppError) as exc:
        run(plan, students)
    assert exc.value.code == ErrorCode.COLUMN_NOT_FOUND and exc.value.details["repairable"]
    assert students.prefix not in exc.value.message


def test_09_append_missing_required_column(students: Caller) -> None:
    plan = create(if_exists="append", columns=[{"name": "roll_no", "pg_type": "integer"}],
                  rows=[["9"]])
    assert code_of(plan, students) == ErrorCode.PLAN_INVALID


def test_10_append_non_writable_type(students: Caller) -> None:
    plan = create(if_exists="append", columns=[{"name": "roll_no", "pg_type": "integer"},
                                               {"name": "name", "pg_type": "text"},
                                               {"name": "notes", "pg_type": "text"}],
                  rows=[["9", "X", "{}"]])
    assert code_of(plan, students) == ErrorCode.PLAN_INVALID


def test_11_replace_own_table(students: Caller) -> None:
    v = run(create(if_exists="replace"), students)
    assert v.statements[0].startswith("DROP TABLE college.students")
    assert v.preview()["destructive"] == ["DROP and recreate college.students"]


def test_12_row_limit(alice: Caller) -> None:
    rows = [[str(i), "n", "2004-01-01"] for i in range(1001)]
    assert code_of(create(rows=rows), alice) == ErrorCode.ROW_LIMIT_EXCEEDED


def test_13_partial_cast_failures_reported(alice: Caller) -> None:
    v = run(create(rows=[["1", "Asha", "2004-05-14"], ["two", "Rahul", "2003-11-02"],
                         ["3", "Meera", "31/31/2005"]]), alice)
    assert v.rows_to_write == 1 and v.cast.failed_row_count == 2
    failed = {(f["row"], f["column"]) for f in v.preview()["failed_rows"]}
    assert failed == {(2, "roll_no"), (3, "dob")}
    assert any("skipped" in w for w in v.warnings)


def test_14_all_rows_fail_is_repairable(alice: Caller) -> None:
    plan = create(columns=[{"name": "roll_no", "pg_type": "integer"},
                           {"name": "name", "pg_type": "integer"},
                           {"name": "dob", "pg_type": "date"}])
    with pytest.raises(AppError) as exc:
        run(plan, alice)
    assert exc.value.code == ErrorCode.TYPE_CAST_FAILED and exc.value.details["repairable"]


def test_15_duplicate_primary_key_in_data(alice: Caller) -> None:
    v = run(create(rows=[["1", "A", None], ["1", "B", None], ["2", "C", None]]), alice)
    assert v.rows_to_write == 2
    assert v.cast.failures[0]["error"] == "duplicate primary key value"


def test_16_row_width_mismatch(alice: Caller) -> None:
    v = run(create(rows=[["1", "A", None], ["2", "B"]]), alice)
    assert v.rows_to_write == 1 and "expected 3" in v.cast.failures[0]["error"]


def test_17_null_primary_key(alice: Caller) -> None:
    v = run(create(rows=[["", "A", None], ["2", "B", None]]), alice)
    assert v.rows_to_write == 1 and v.cast.failures[0]["column"] == "roll_no"


def test_18_no_primary_key_warns(alice: Caller) -> None:
    v = run(create(columns=[{"name": "a", "pg_type": "text"}], rows=[["x"]]), alice)
    assert any("no primary key" in w for w in v.warnings)


def test_19_cannot_name_another_users_schema(students: Caller, bob: Caller) -> None:
    """Writing alice's physical name only reaches bob's own (nonexistent) schema."""
    plan = InsertRowsPlan.model_validate({
        "schema": f"{students.prefix}college", "table": "students",
        "columns": ["roll_no", "name"], "rows": [["99", "Mallory"]]})
    assert code_of(plan, bob) == ErrorCode.TABLE_NOT_FOUND


# ---- insert ---------------------------------------------------------------- #
def insert(**kw: Any) -> InsertRowsPlan:
    data = {"schema": "college", "table": "students", "columns": ["roll_no", "name"],
            "rows": [["10", "Zed"]], **kw}
    return InsertRowsPlan.model_validate(data)


def test_20_insert_ok(students: Caller) -> None:
    v = run(insert(), students)
    assert v.cast.rows == [[10, "Zed"]] and "INSERT INTO college.students" in v.statements[0]


def test_21_insert_missing_table(alice: Caller) -> None:
    assert code_of(insert(), alice) == ErrorCode.TABLE_NOT_FOUND


def test_22_insert_into_other_users_table(students: Caller, bob: Caller) -> None:
    assert code_of(insert(), bob) == ErrorCode.TABLE_NOT_FOUND


def test_23_insert_unknown_column(students: Caller) -> None:
    assert code_of(insert(columns=["roll_no", "nickname"]), students) == ErrorCode.COLUMN_NOT_FOUND


def test_24_insert_missing_not_null(students: Caller) -> None:
    assert code_of(insert(columns=["roll_no"], rows=[["11"]]), students) == ErrorCode.PLAN_INVALID


# ---- update / delete ------------------------------------------------------- #
def update(**kw: Any) -> UpdateRowsPlan:
    data = {"schema": "college", "table": "students", "set": [{"column": "age", "value": 30}],
            "where": [{"column": "roll_no", "op": "=", "value": 1}], **kw}
    return UpdateRowsPlan.model_validate(data)


def delete(**kw: Any) -> DeleteRowsPlan:
    data = {"schema": "college", "table": "students",
            "where": [{"column": "age", "op": ">=", "value": 20}], **kw}
    return DeleteRowsPlan.model_validate(data)


def test_25_update_counts_affected_rows(students: Caller) -> None:
    v = run(update(), students)
    assert v.affected_rows == 1 and "UPDATE college.students" in v.statements[0]


def test_26_update_without_where_needs_explicit_all_rows(students: Caller) -> None:
    assert code_of(update(where=[]), students) == ErrorCode.DESTRUCTIVE_OPERATION_BLOCKED
    v = run(update(where=[], all_rows=True), students)
    assert v.affected_rows == 3 and v.preview()["destructive"]


def test_27_update_bad_value_type(students: Caller) -> None:
    with pytest.raises(AppError) as exc:
        run(update(set=[{"column": "age", "value": "old"}]), students)
    assert exc.value.code == ErrorCode.TYPE_CAST_FAILED and exc.value.details["repairable"]


def test_28_update_not_null_to_null(students: Caller) -> None:
    assert code_of(update(set=[{"column": "name", "value": None}]), students) \
        == ErrorCode.TYPE_CAST_FAILED


def test_29_delete_own_rows(students: Caller) -> None:
    assert run(delete(), students).affected_rows == 2


def test_30_delete_without_where_needs_explicit_all_rows(students: Caller) -> None:
    assert code_of(delete(where=[]), students) == ErrorCode.DESTRUCTIVE_OPERATION_BLOCKED
    assert run(delete(where=[], all_rows=True), students).affected_rows == 3


def test_31_update_delete_other_users_table(students: Caller, bob: Caller) -> None:
    assert code_of(update(), bob) == ErrorCode.TABLE_NOT_FOUND
    assert code_of(delete(), bob) == ErrorCode.TABLE_NOT_FOUND


def test_32_where_unknown_column(students: Caller) -> None:
    assert code_of(delete(where=[{"column": "zzz", "op": "=", "value": 1}]), students) \
        == ErrorCode.COLUMN_NOT_FOUND


def test_33_where_bad_value(students: Caller) -> None:
    assert code_of(delete(where=[{"column": "age", "op": "=", "value": "twenty"}]), students) \
        == ErrorCode.TYPE_CAST_FAILED
    assert code_of(delete(where=[{"column": "age", "op": "like", "value": "2%"}]), students) \
        == ErrorCode.TYPE_CAST_FAILED


def test_34_where_in_and_null(students: Caller) -> None:
    v = run(delete(where=[{"column": "roll_no", "op": "in", "value": [1, 2, 99]},
                          {"column": "dob", "op": "is_not_null"}]), students)
    assert v.affected_rows == 2


def test_35_identifier_smuggled_past_pydantic_is_rejected(students: Caller) -> None:
    plan = insert().model_copy(update={"table": 'students"; DROP TABLE x; --'})
    assert code_of(plan, students) == ErrorCode.INVALID_IDENTIFIER
    plan = insert().model_copy(update={"schema_name": "private"})
    assert code_of(plan, students) == ErrorCode.TABLE_NOT_FOUND  # u<id>_private, not "private"
