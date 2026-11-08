"""Pydantic-level plan validation (no database)."""
import pytest
from pydantic import ValidationError

from app.schemas.plans import (
    CreateTablePlan,
    DeleteRowsPlan,
    IngestDraft,
    InsertRowsPlan,
    UpdateRowsPlan,
    dump_plan,
    parse_plan,
)


def test_create_plan_normalizes_names() -> None:
    plan = CreateTablePlan.model_validate({
        "schema": "College", "table": "Student List",
        "columns": [{"name": "Roll No", "pg_type": "INT", "primary_key": True},
                    {"name": "Order", "pg_type": "text"}],
    })
    assert (plan.schema_name, plan.table) == ("college", "student_list")
    assert plan.column_names == ["roll_no", "order_col"]
    assert plan.columns[0].pg_type == "integer" and plan.columns[0].nullable is False


@pytest.mark.parametrize(
    "payload",
    [
        {"schema": "pg_catalog", "table": "t", "columns": [{"name": "a", "pg_type": "text"}]},
        {"schema": "college", "table": "select", "columns": [{"name": "a", "pg_type": "text"}]},
        {"schema": "college", "table": "t", "columns": []},
        {"schema": "college", "table": "t", "columns": [{"name": "a", "pg_type": "json"}]},
        {"schema": "college", "table": "t",
         "columns": [{"name": "A", "pg_type": "text"}, {"name": "a", "pg_type": "text"}]},
        {"schema": "college", "table": "t", "columns": [{"name": "a", "pg_type": "text"}],
         "if_exists": "drop"},
        {"schema": "college", "table": "t", "columns": [{"name": "a", "pg_type": "text"}],
         "sql": "DROP TABLE x"},  # extra keys are forbidden
        {"schema": "college", "table": "t", "columns": [{"name": "a", "pg_type": "text"}],
         "rows": [[{"nested": "object"}]]},
    ],
)
def test_create_plan_rejects(payload: dict) -> None:
    with pytest.raises(ValidationError):
        CreateTablePlan.model_validate(payload)


def test_conditions_shapes() -> None:
    ok = DeleteRowsPlan.model_validate({"schema": "college", "table": "t",
                                        "where": [{"column": "id", "op": "in", "value": [1, 2]}]})
    assert ok.where[0].value == [1, 2]
    for bad in ({"column": "id", "op": "in", "value": 1},
                {"column": "id", "op": "=", "value": None},
                {"column": "id", "op": "=", "value": [1]},
                {"column": "id", "op": "; drop", "value": 1}):
        with pytest.raises(ValidationError):
            DeleteRowsPlan.model_validate({"schema": "college", "table": "t", "where": [bad]})


def test_update_and_insert_models() -> None:
    with pytest.raises(ValidationError):
        UpdateRowsPlan.model_validate({"schema": "college", "table": "t", "set": []})
    with pytest.raises(ValidationError):
        UpdateRowsPlan.model_validate({"schema": "college", "table": "t",
                                       "set": [{"column": "a", "value": 1}, {"column": "A", "value": 2}]})
    with pytest.raises(ValidationError):
        InsertRowsPlan.model_validate({"schema": "college", "table": "t", "columns": ["a"], "rows": []})


def test_ingest_draft_unique_sources() -> None:
    with pytest.raises(ValidationError):
        IngestDraft.model_validate({"schema": "college", "table": "t", "columns": [
            {"source_index": 0, "name": "a", "pg_type": "text"},
            {"source_index": 0, "name": "b", "pg_type": "text"}]})


def test_plan_round_trip() -> None:
    plan = InsertRowsPlan.model_validate({"schema": "college", "table": "t", "columns": ["a"],
                                          "rows": [["1"], [None]]})
    again = parse_plan(dump_plan(plan))
    assert again == plan and dump_plan(plan)["schema"] == "college"
