"""Extraction result shared by the PDF / CSV / XLSX readers."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

Cell = str | None


@dataclass
class ExtractedTable:
    header: list[str]
    rows: list[list[Cell]]
    page: int | None = None  # first page (PDF) or None
    method: str = "table"  # "table" (ruled/aligned table), "text_blocks", "csv", "xlsx"

    @property
    def width(self) -> int:
        return len(self.header)

    def summary(self, sample: int = 3) -> dict[str, Any]:
        return {"page": self.page, "header": self.header, "row_count": len(self.rows),
                "method": self.method, "sample_rows": self.rows[:sample]}


@dataclass
class Extraction:
    source: str  # pdf | csv | xlsx
    pages: int
    tables: list[ExtractedTable] = field(default_factory=list)
    raw_text_excerpt: str = ""
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "pages": self.pages,
            "tables": [dict(index=i, **t.summary()) for i, t in enumerate(self.tables)],
            "raw_text_excerpt": self.raw_text_excerpt[:500],
            "warnings": self.warnings,
        }


def clean_cell(value: object) -> Cell:
    if value is None:
        return None
    text = " ".join(str(value).replace("\x00", "").split())
    return text or None


def normalize_table(
    raw_rows: Sequence[Sequence[object] | None],
) -> tuple[list[str], list[list[Cell]]] | None:
    """First non-empty row is the header; pads/trims rows to the header width and
    drops fully empty rows and columns. Returns None if it is not a usable table."""
    rows = [[clean_cell(c) for c in r] for r in raw_rows if r is not None]
    rows = [r for r in rows if any(c is not None for c in r)]
    if len(rows) < 2:
        return None
    width = max(len(r) for r in rows)
    rows = [r + [None] * (width - len(r)) for r in rows]
    keep = [i for i in range(width) if any(r[i] is not None for r in rows)]
    rows = [[r[i] for i in keep] for r in rows]
    if len(keep) < 2:
        return None
    header = [h or f"column_{i + 1}" for i, h in enumerate(rows[0])]
    return header, rows[1:]
