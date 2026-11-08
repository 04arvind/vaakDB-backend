import uuid
from datetime import date, datetime
from decimal import Decimal

import pytest

from app.sql.parser import parse_pg_type
from app.sql.validator import CastError, ColumnSpec, cast_rows, cast_value, to_jsonable

T = parse_pg_type


@pytest.mark.parametrize(
    ("value", "type_", "expected"),
    [
        ("42", "integer", 42), ("1,234", "integer", 1234), ("12.0", "integer", 12),
        (7, "bigint", 7), ("9223372036854775807", "bigint", 9223372036854775807),
        ("8.75", "numeric(4,2)", Decimal("8.75")), ("₹1,200.5", "numeric(10,2)", Decimal("1200.50")),
        ("(5)", "numeric", Decimal("-5")), ("yes", "boolean", True), ("F", "boolean", False),
        ("2004-05-14", "date", date(2004, 5, 14)), ("14/05/2004", "date", date(2004, 5, 14)),
        ("14 May 2004", "date", date(2004, 5, 14)), ("05/31/2004", "date", date(2004, 5, 31)),
        ("2024-01-02 03:04:05", "timestamp", datetime(2024, 1, 2, 3, 4, 5)),
        ("2024-01-02", "timestamp", datetime(2024, 1, 2)),
        (" hello ", "text", "hello"), ("abc", "varchar(3)", "abc"),
        ("", "integer", None), ("N/A", "date", None), (None, "text", None),
    ],
)
def test_cast_ok(value: object, type_: str, expected: object) -> None:
    assert cast_value(value, T(type_)) == expected


@pytest.mark.parametrize(
    ("value", "type_"),
    [
        ("abc", "integer"), ("12.5", "integer"), ("99999999999", "integer"), (True, "integer"),
        ("1e200000", "numeric"), ("123.45", "numeric(4,2)"), ("maybe", "boolean"),
        ("2024-13-45", "date"), ("05/06/2004 noon", "timestamp"), ("abcd", "varchar(3)"),
        ("not-a-uuid", "uuid"), ("inf", "numeric"),
    ],
)
def test_cast_fails(value: object, type_: str) -> None:
    with pytest.raises(CastError):
        cast_value(value, T(type_))


def test_uuid_and_json() -> None:
    u = uuid.uuid4()
    assert cast_value(str(u), T("uuid")) == u
    assert to_jsonable(Decimal("1.50")) == "1.50"
    assert to_jsonable(date(2024, 1, 2)) == "2024-01-02"


def test_cast_rows_reports_failures_and_not_null() -> None:
    cols = [ColumnSpec("id", T("integer"), nullable=False), ColumnSpec("score", T("numeric(4,1)"))]
    report = cast_rows(cols, [["1", "9.5"], ["x", "1"], ["", "2"], ["4", "oops"], ["5"]])
    assert report.rows == [[1, Decimal("9.5")]]
    assert report.row_numbers == [1]
    assert report.failed_row_count == 4
    errors = {(f["row"], f["column"]) for f in report.failures}
    assert {(2, "id"), (3, "id"), (4, "score"), (5, None)} <= errors
