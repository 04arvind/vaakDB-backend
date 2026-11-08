"""PDF -> tables, with pdfplumber.

1. Ruled/aligned tables per page (pdfplumber's table finder).
2. If none are found: "text blocks", i.e. runs of consecutive lines that split
   into the same number (>= 2) of columns on 2+ spaces, read with layout=True.
3. If the PDF has no extractable text at all it is scanned: SCANNED_PDF_UNSUPPORTED
   (OCR is out of scope).

Tables that continue on the next page with the same header are merged.
Everything returned here is untrusted data; it is never executed or used as
instructions.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from app.core.exceptions import AppError, ErrorCode
from app.services.ingest.models import ExtractedTable, Extraction, normalize_table

log = logging.getLogger(__name__)

_SPLIT = re.compile(r"\s{2,}|\t")
MIN_TEXT_BLOCK_ROWS = 2  # data rows, besides the header


def _block_to_table(block: list[list[str]], page_no: int) -> ExtractedTable | None:
    if len(block) < MIN_TEXT_BLOCK_ROWS + 1:
        return None
    normalized = normalize_table(block)
    if not normalized:
        return None
    header, rows = normalized
    return ExtractedTable(header=header, rows=rows, page=page_no, method="text_blocks")


def _text_blocks(page_texts: list[tuple[int, str]]) -> list[ExtractedTable]:
    """Runs of >= 3 consecutive lines that split into the same number (>= 2) of columns."""
    tables: list[ExtractedTable] = []
    for page_no, text in page_texts:
        blocks: list[list[list[str]]] = [[]]
        for line in text.splitlines():
            cells = [c for c in _SPLIT.split(line.strip()) if c]
            current = blocks[-1]
            if len(cells) >= 2 and (not current or len(cells) == len(current[0])):
                current.append(cells)
            else:
                blocks.append([cells] if len(cells) >= 2 else [])
        tables.extend(t for b in blocks if (t := _block_to_table(b, page_no)) is not None)
    return tables


def _merge_continuations(tables: list[ExtractedTable]) -> list[ExtractedTable]:
    merged: list[ExtractedTable] = []
    for t in tables:
        prev = merged[-1] if merged else None
        if prev is not None and prev.method == t.method and prev.header == t.header:
            prev.rows.extend(t.rows)
        elif (prev is not None and prev.method == t.method and t.width == prev.width
              and prev.page is not None and t.page == prev.page + 1
              and _looks_like_data(t.header)):
            # page break without a repeated header: the "header" is a data row
            prev.rows.append(list(t.header))
            prev.rows.extend(t.rows)
        else:
            merged.append(t)
    return merged


def _looks_like_data(header: list[str]) -> bool:
    """A header row of mostly numbers is really the first data row of a continuation."""
    numeric = sum(1 for h in header if re.fullmatch(r"[-+]?[\d.,/:]+", h or ""))
    return numeric >= max(1, len(header) // 2)


def extract_pdf(path: Path, *, max_pages: int = 50) -> Extraction:
    import pdfplumber

    try:
        pdf = pdfplumber.open(path)
    except Exception as exc:  # encrypted, truncated, not really a PDF
        log.info("PDF could not be opened: %s", type(exc).__name__)
        raise AppError(ErrorCode.FILE_UNREADABLE,
                       "The PDF could not be read (corrupt or encrypted).") from None

    warnings: list[str] = []
    tables: list[ExtractedTable] = []
    page_texts: list[tuple[int, str]] = []
    total_chars = 0
    with pdf:
        page_count = len(pdf.pages)
        if page_count > max_pages:
            warnings.append(f"Only the first {max_pages} of {page_count} pages were read.")
        for page_no, page in enumerate(pdf.pages[:max_pages], start=1):
            try:
                plain = page.extract_text() or ""
                total_chars += len(plain.strip())
                page_texts.append((page_no, page.extract_text(layout=True) or ""))
                for raw in page.extract_tables():
                    normalized = normalize_table(raw)
                    if normalized:
                        header, rows = normalized
                        tables.append(ExtractedTable(header=header, rows=rows, page=page_no))
            except Exception as exc:
                log.warning("PDF page %d failed: %s", page_no, type(exc).__name__)
                warnings.append(f"Page {page_no} could not be read and was skipped.")
            finally:
                if hasattr(page, "flush_cache"):
                    page.flush_cache()

    if total_chars == 0:
        raise AppError(ErrorCode.SCANNED_PDF_UNSUPPORTED)

    if not tables:
        tables = _text_blocks(page_texts)
        if tables:
            warnings.append("No ruled table was found; columns were read from aligned text.")

    excerpt = "\n".join(t for _, t in page_texts)
    excerpt = re.sub(r"[ \t]{2,}", "  ", excerpt)
    return Extraction(
        source="pdf",
        pages=page_count,
        tables=_merge_continuations(tables),
        raw_text_excerpt=excerpt[:2000],
        warnings=warnings,
    )
