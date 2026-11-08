"""Metadata operations: every user sees only their own workspace."""
from __future__ import annotations

from typing import Any

import pytest

from app.core.exceptions import AppError, ErrorCode
from app.security.authorization import Caller
from app.services import metadata_ops


@pytest.fixture
def alice(caller_for: Any, clean_target: Any, ws_exec: Any) -> Caller:
    a = caller_for()
    ws_exec(a,
            "CREATE TABLE {s}.students (roll_no int PRIMARY KEY, name text NOT NULL, "
            "age int, api_token text)",
            "INSERT INTO {s}.students SELECT g, 'student ' || g, 18 + g % 5, 'tok' || g "
            "FROM generate_series(1, 25) g",
            "CREATE TABLE {s}.courses (id int PRIMARY KEY, title text)",
            "CREATE TABLE {s}.enrollments (student int REFERENCES {s}.students, "
            "course int REFERENCES {s}.courses)",
            "ANALYZE {s}.students")
    ws_exec(a, "CREATE TABLE {s}.staff (id int PRIMARY KEY, name text)", schema="hr")
    return a


@pytest.fixture
def bob(caller_for: Any, ws_exec: Any) -> Caller:
    b = caller_for()
    ws_exec(b, "CREATE TABLE {s}.notes (id int PRIMARY KEY)", schema="main")
    return b


def test_list_schemas_only_own(alice: Caller, bob: Caller) -> None:
    assert [i["schema"] for i in metadata_ops.list_schemas(alice)["data"]["items"]] == ["college", "hr"]
    assert [i["schema"] for i in metadata_ops.list_schemas(bob)["data"]["items"]] == ["main"]


def test_empty_workspace(caller_for: Any, clean_target: Any) -> None:
    env = metadata_ops.list_schemas(caller_for())
    assert env["data"]["items"] == [] and "empty" in env["message"]


def test_list_tables(alice: Caller, bob: Caller) -> None:
    names = {i["table"] for i in metadata_ops.list_tables(alice, "college")["data"]["items"]}
    assert names == {"students", "courses", "enrollments"}
    with pytest.raises(AppError) as exc:  # bob has no "college" schema of his own
        metadata_ops.list_tables(bob, "college")
    assert exc.value.code == ErrorCode.SCHEMA_NOT_FOUND


@pytest.mark.parametrize("schema", ["pg_catalog", "information_schema"])
def test_system_schemas_rejected(alice: Caller, schema: str) -> None:
    with pytest.raises(AppError) as exc:
        metadata_ops.list_tables(alice, schema)
    assert exc.value.code in (ErrorCode.FORBIDDEN_SCHEMA, ErrorCode.INVALID_IDENTIFIER)


@pytest.mark.parametrize("schema", ["public", "private"])
def test_shared_schemas_are_not_reachable(alice: Caller, schema: str) -> None:
    """'public' / 'private' mean the user's own u<id>_public, never the real ones."""
    with pytest.raises(AppError) as exc:
        metadata_ops.list_tables(alice, schema)
    assert exc.value.code == ErrorCode.SCHEMA_NOT_FOUND


def test_other_users_table_looks_exactly_like_missing(alice: Caller, bob: Caller) -> None:
    errors = []
    for schema, name in ((f"{alice.prefix}college", "students"), ("college", "students"),
                         ("college", "does_not_exist")):
        with pytest.raises(AppError) as exc:
            metadata_ops.describe_table(bob, schema, name)
        errors.append(exc.value.code)
    assert set(errors) == {ErrorCode.TABLE_NOT_FOUND}


def test_describe(alice: Caller) -> None:
    env = metadata_ops.describe_table(alice, "college", "enrollments")
    cols = {c["name"]: c for c in env["data"]["columns"]}
    assert cols["student"]["references"] == "college.students.roll_no"  # logical name, own data
    d = metadata_ops.describe_table(alice, "college", "students")["data"]
    assert d["primary_key"] == ["roll_no"] and d["row_count"] == 25
    assert len(d["sample_rows"]) == 5 and d["sample_rows"][0]["api_token"] == "***"


def test_rows_pagination_and_sort_whitelist(alice: Caller) -> None:
    page = metadata_ops.table_rows(alice, "college", "students", page=2, page_size=10,
                                   sort="roll_no", direction="desc")["data"]
    assert [r["roll_no"] for r in page["rows"]][:2] == [15, 14]
    assert page["total_rows"] == 25 and page["has_more"]
    with pytest.raises(AppError) as exc:
        metadata_ops.table_rows(alice, "college", "students", sort="roll_no; DROP TABLE x")
    assert exc.value.code == ErrorCode.COLUMN_NOT_FOUND


def test_csv_export(alice: Caller, bob: Caller, ws_exec: Any) -> None:
    ws_exec(alice, "INSERT INTO {s}.courses VALUES (1, '=HYPERLINK(\"x\")')")
    text = "".join(metadata_ops.export_csv(alice, "college", "courses"))
    assert text.splitlines()[0] == "id,title" and "'=HYPERLINK" in text
    body = "".join(metadata_ops.export_csv(alice, "college", "students"))
    assert "tok1" not in body and body.count("\n") == 26
    with pytest.raises(AppError) as exc:
        metadata_ops.export_csv(bob, "college", "students")
    assert exc.value.code == ErrorCode.TABLE_NOT_FOUND
