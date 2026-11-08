import pytest

from app.core.exceptions import AppError, ErrorCode
from app.db.urls import mask_url, normalize_pg_url
from app.sql.parser import (
    canonical_type,
    dedupe_identifiers,
    is_valid_identifier,
    normalize_identifier,
    parse_pg_type,
    validate_identifier,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Roll No", "roll_no"),
        ("Date of Birth", "date_of_birth"),
        ("  GPA  ", "gpa"),
        ("Marks (%)", "marks_pct"),
        ("Student-Name", "student_name"),
        ("e-mail.address", "e_mail_address"),
        ("2024 Marks", "_2024_marks"),
        ("Naïve Café", "naive_cafe"),
        ("a__b___c", "a_b_c"),
        ("Phone #", "phone_no"),
        ("x" * 100, "x" * 63),
    ],
)
def test_normalize_identifier(raw: str, expected: str) -> None:
    assert normalize_identifier(raw, "column") == expected


@pytest.mark.parametrize("raw", ["", "   ", "!!!", "@$^", None])
def test_normalize_rejects_empty(raw: object) -> None:
    with pytest.raises(AppError) as exc:
        normalize_identifier(raw, "column")
    assert exc.value.code == ErrorCode.INVALID_IDENTIFIER


def test_reserved_words_rejected_or_suffixed() -> None:
    with pytest.raises(AppError):
        normalize_identifier("Order", "table")
    assert normalize_identifier("Order", "column", on_reserved="suffix") == "order_col"
    assert normalize_identifier("user", "column", on_reserved="suffix") == "user_col"
    assert normalize_identifier("ctid", "column", on_reserved="suffix") == "ctid_col"


def test_schema_rules() -> None:
    with pytest.raises(AppError):
        normalize_identifier("pg_catalog", "schema")
    with pytest.raises(AppError):
        normalize_identifier("select", "schema", on_reserved="suffix")  # schemas never suffixed
    assert normalize_identifier("College", "schema") == "college"


@pytest.mark.parametrize("name", ["students", "_x", "a1", "a" * 63])
def test_valid(name: str) -> None:
    assert validate_identifier(name, "table") == name
    assert is_valid_identifier(name, "table")


@pytest.mark.parametrize("name", ["Students", "1abc", "a-b", "a b", "a" * 64, "select", 'x";drop', ""])
def test_invalid(name: str) -> None:
    assert not is_valid_identifier(name, "table")
    with pytest.raises(AppError):
        validate_identifier(name, "table")


def test_dedupe_identifiers() -> None:
    assert dedupe_identifiers(["name", "name", "name_2"]) == ["name", "name_3", "name_2"]
    assert dedupe_identifiers(["a", "b"]) == ["a", "b"]


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("INT", "integer"), ("int4", "integer"), ("bigint", "bigint"), ("Text", "text"),
        ("varchar(40)", "varchar(40)"), ("character varying(10)", "varchar(10)"),
        ("varchar", "varchar(255)"), ("decimal(10, 2)", "numeric(10,2)"), ("numeric", "numeric"),
        ("float", "numeric"), ("bool", "boolean"), ("DATE", "date"),
        ("timestamp without time zone", "timestamp"), ("uuid", "uuid"),
    ],
)
def test_type_whitelist(raw: str, canonical: str) -> None:
    assert canonical_type(raw) == canonical


@pytest.mark.parametrize("raw", ["json", "jsonb", "bytea", "text[]", "int; drop table x",
                                 "varchar(0)", "numeric(5,9)", "money(2)", "serial primary key"])
def test_type_rejected(raw: str) -> None:
    with pytest.raises(AppError) as exc:
        parse_pg_type(raw)
    assert exc.value.code == ErrorCode.PLAN_INVALID


def test_url_with_at_sign_in_password() -> None:
    url = normalize_pg_url('"postgresql://postgres:Arvind@2005@localhost:5432/vaakDB"')
    assert url == "postgresql+psycopg://postgres:Arvind%402005@localhost:5432/vaakDB"
    assert normalize_pg_url(url) == url  # idempotent
    assert mask_url(url) == "postgresql+psycopg://postgres:***@localhost:5432/vaakDB"
