"""Value casting: turn raw cell values (strings from files or the LLM) into
typed Python values for the whitelisted column types.

Casting is strict and deterministic. It runs twice: once at preview time (to
report rows that fail) and again at execution time, so the executor only ever
binds values it has validated itself.

Dates: ISO (2024-03-15) first, then day-first formats (15/03/2024, 15-03-2024,
15 Mar 2024). Month-first (03/15/2024) is only accepted when the day is > 12.
"""
from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from app.sql.parser import PgType

NULL_TOKENS = frozenset({"", "null", "none", "nil", "na", "n/a", "nan", "-", "--"})
TRUE_TOKENS = frozenset({"true", "t", "yes", "y", "1", "on"})
FALSE_TOKENS = frozenset({"false", "f", "no", "n", "0", "off"})
INT32 = (-(2**31), 2**31 - 1)
INT64 = (-(2**63), 2**63 - 1)
MAX_TEXT_CHARS = 1_000_000

_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %b %Y",
                 "%d %B %Y", "%b %d %Y", "%B %d %Y", "%d-%b-%Y", "%d/%m/%y", "%d-%m-%y")
_TIME_SUFFIXES = (" %H:%M:%S", " %H:%M", "T%H:%M:%S", "T%H:%M")
_NUMBER_NOISE = re.compile(r"[,\s_]|^[$€£₹¥]|[$€£₹¥]$")


class CastError(ValueError):
    """A value does not fit its column type. The message is safe to show users."""


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip().lower() in NULL_TOKENS


def _number_text(value: Any) -> str:
    if isinstance(value, bool):
        raise CastError("expected a number, got a boolean")
    text = str(value).strip()
    negative = text.startswith("(") and text.endswith(")")  # accounting negatives
    text = _NUMBER_NOISE.sub("", text.strip("()"))
    return "-" + text if negative else text


def _to_int(value: Any, bounds: tuple[int, int], label: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        number = value
    else:
        text = _number_text(value)
        try:
            dec = Decimal(text)
        except InvalidOperation:
            raise CastError(f"{str(value)[:40]!r} is not a whole number") from None
        if not dec.is_finite() or dec != dec.to_integral_value():
            raise CastError(f"{str(value)[:40]!r} is not a whole number")
        number = int(dec)
    if not bounds[0] <= number <= bounds[1]:
        raise CastError(f"{number} is out of range for {label}")
    return number


def _to_decimal(value: Any, t: PgType) -> Decimal:
    try:
        dec = Decimal(_number_text(value))
    except InvalidOperation:
        raise CastError(f"{str(value)[:40]!r} is not a number") from None
    if not dec.is_finite():
        raise CastError("infinite or NaN numbers are not allowed")
    if dec != 0 and not -16383 <= dec.adjusted() <= 131071:  # Postgres numeric limits
        raise CastError(f"{str(value)[:40]} is out of range for numeric")
    if t.precision is not None:
        scale = t.scale or 0
        dec = dec.quantize(Decimal(1).scaleb(-scale)) if scale else dec.quantize(Decimal(1))
        int_digits = len(dec.as_tuple().digits) - scale if dec != 0 else 0
        if int_digits > t.precision - scale:
            raise CastError(f"{value} does not fit numeric({t.precision},{scale})")
    return dec


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in TRUE_TOKENS:
        return True
    if text in FALSE_TOKENS:
        return False
    raise CastError(f"{str(value)[:40]!r} is not true/false")


def _to_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = re.sub(r"\s+", " ", str(value).strip().replace(",", ""))
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    try:  # unambiguous month-first, e.g. 03/15/2024
        parsed = datetime.strptime(text, "%m/%d/%Y").date()
        if parsed.day > 12:
            return parsed
    except ValueError:
        pass
    raise CastError(f"{str(value)[:40]!r} is not a date (use YYYY-MM-DD)")


def _to_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except ValueError:
        pass
    for base in _DATE_FORMATS:
        for suffix in _TIME_SUFFIXES:
            try:
                return datetime.strptime(text, base + suffix)
            except ValueError:
                continue
    try:
        d = _to_date(text)
        return datetime(d.year, d.month, d.day)
    except CastError:
        raise CastError(f"{str(value)[:40]!r} is not a timestamp (use YYYY-MM-DD HH:MM:SS)") from None


def cast_value(value: Any, t: PgType) -> Any:
    """Return a typed value (None for empty cells) or raise CastError."""
    if _is_null(value):
        return None
    if t.name in ("text", "varchar"):
        text = value if isinstance(value, str) else str(value)
        text = text.strip().replace("\x00", "")
        limit = t.length if t.name == "varchar" and t.length else MAX_TEXT_CHARS
        if len(text) > limit:
            raise CastError(f"text is {len(text)} characters, longer than {limit}")
        return text
    if t.name == "integer":
        return _to_int(value, INT32, "integer")
    if t.name == "bigint":
        return _to_int(value, INT64, "bigint")
    if t.name == "numeric":
        return _to_decimal(value, t)
    if t.name == "boolean":
        return _to_bool(value)
    if t.name == "date":
        return _to_date(value)
    if t.name == "timestamp":
        return _to_timestamp(value)
    if t.name == "uuid":
        try:
            return uuid.UUID(str(value).strip())
        except ValueError:
            raise CastError(f"{str(value)[:40]!r} is not a UUID") from None
    raise CastError(f"unsupported type {t}")


def to_jsonable(value: Any) -> Any:
    """Typed value -> JSON-safe value (stored in plan_json / sent to the frontend)."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (Decimal, uuid.UUID)):
        return str(value)
    return value


@dataclass
class ColumnSpec:
    name: str
    type: PgType
    nullable: bool = True


@dataclass
class CastReport:
    rows: list[list[Any]] = field(default_factory=list)  # typed values, good rows only
    row_numbers: list[int] = field(default_factory=list)  # 1-based source row numbers kept
    failures: list[dict[str, Any]] = field(default_factory=list)
    failed_row_count: int = 0

    MAX_FAILURES_REPORTED = 50


def cast_rows(columns: list[ColumnSpec], rows: list[list[Any]]) -> CastReport:
    """Cast every value. Rows with any failing value are left out and reported."""
    report = CastReport()
    width = len(columns)
    for index, row in enumerate(rows, start=1):
        problems: list[dict[str, Any]] = []
        if len(row) != width:
            problems.append({"row": index, "column": None, "value": None,
                             "error": f"row has {len(row)} values, expected {width}"})
            typed: list[Any] = []
        else:
            typed = []
            for col, raw in zip(columns, row, strict=True):
                try:
                    value = cast_value(raw, col.type)
                    if value is None and not col.nullable:
                        raise CastError("a value is required (column is NOT NULL)")
                    typed.append(value)
                except CastError as exc:
                    problems.append({"row": index, "column": col.name,
                                     "value": None if raw is None else str(raw)[:100],
                                     "error": str(exc)})
        if problems:
            report.failed_row_count += 1
            room = CastReport.MAX_FAILURES_REPORTED - len(report.failures)
            report.failures.extend(problems[:max(room, 0)])
        else:
            report.rows.append(typed)
            report.row_numbers.append(index)
    return report
