"""CSV and XLSX readers (standard library only).

XLSX: the first worksheet is read straight from the OOXML zip. Shared strings,
inline strings, booleans and numbers are supported; numbers formatted as dates
are converted to ISO dates. Formulas give their cached value.
"""
from __future__ import annotations

import csv
import io
import re
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

from app.core.exceptions import AppError, ErrorCode
from app.services.ingest.models import ExtractedTable, Extraction, normalize_table

MAX_XLSX_UNCOMPRESSED = 200 * 1024 * 1024  # zip-bomb guard
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
_BUILTIN_DATE_FORMATS = set(range(14, 23)) | {45, 46, 47}
_EXCEL_EPOCH = datetime(1899, 12, 30)


def _xml(data: bytes) -> ET.Element:
    """Parse one OOXML part. DTDs/entities never appear in valid XLSX, so they are
    rejected outright (no entity expansion, no external entities)."""
    head = data[:4096].upper()
    if b"<!DOCTYPE" in head or b"<!ENTITY" in data.upper():
        raise AppError(ErrorCode.FILE_UNREADABLE, "The XLSX file contains unsupported XML.")
    return ET.fromstring(data)  # noqa: S314 - DTD/entities rejected above


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _guess_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        pass
    lines = [line for line in sample.split("\n") if line.strip()][:50]
    best, best_score = ",", 0.0
    for delim in (",", ";", "\t", "|"):
        counts = [line.count(delim) for line in lines]
        if not counts or counts[0] == 0:
            continue
        consistent = sum(1 for c in counts if c == counts[0]) / len(counts)
        score = consistent * counts[0]
        if consistent >= 0.6 and score > best_score:
            best, best_score = delim, score
    return best


def extract_csv(path: Path) -> Extraction:
    text = _decode(path.read_bytes())
    if "\x00" in text:
        raise AppError(ErrorCode.FILE_UNREADABLE, "The CSV file contains binary data.")
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # the sniffer fails on CRLF files
    raw = list(csv.reader(io.StringIO(text), delimiter=_guess_delimiter(text[:20_000])))
    normalized = normalize_table(raw)
    if not normalized:
        raise AppError(ErrorCode.NO_TABLE_FOUND_IN_FILE, "The CSV file has no header and data rows.")
    header, rows = normalized
    return Extraction(source="csv", pages=1,
                      tables=[ExtractedTable(header=header, rows=rows, method="csv")],
                      raw_text_excerpt=text[:2000])


# --------------------------------------------------------------------------- #
def _col_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref or "")
    if not letters:
        return 0
    n = 0
    for ch in letters.group(0):
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _date_styles(zf: zipfile.ZipFile) -> set[int]:
    """Indexes of cell styles (cellXfs) that display numbers as dates."""
    try:
        root = _xml(zf.read("xl/styles.xml"))
    except KeyError:
        return set()
    custom: set[int] = set()
    for fmt in root.findall("m:numFmts/m:numFmt", _NS):
        code = re.sub(r'"[^"]*"|\[[^\]]*\]', "", fmt.get("formatCode", "").lower())
        if re.search(r"[dy]|m{3,}", code) and "h" not in code:
            custom.add(int(fmt.get("numFmtId", "0")))
    out = set()
    for i, xf in enumerate(root.findall("m:cellXfs/m:xf", _NS)):
        fid = int(xf.get("numFmtId", "0"))
        if fid in _BUILTIN_DATE_FORMATS or fid in custom:
            out.add(i)
    return out


def _first_sheet_path(zf: zipfile.ZipFile) -> str:
    wb = _xml(zf.read("xl/workbook.xml"))
    sheet = wb.find("m:sheets/m:sheet", _NS)
    if sheet is None:
        raise AppError(ErrorCode.NO_TABLE_FOUND_IN_FILE, "The workbook has no sheets.")
    rel_id = sheet.get(_REL_NS)
    rels = _xml(zf.read("xl/_rels/workbook.xml.rels"))
    for rel in rels:
        if rel.get("Id") == rel_id:
            target = rel.get("Target", "").lstrip("/")
            return target if target.startswith("xl/") else f"xl/{target}"
    return "xl/worksheets/sheet1.xml"


def extract_xlsx(path: Path) -> Extraction:
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise AppError(ErrorCode.FILE_UNREADABLE, "The XLSX file could not be read.") from None
    with zf:
        if sum(i.file_size for i in zf.infolist()) > MAX_XLSX_UNCOMPRESSED:
            raise AppError(ErrorCode.FILE_UNREADABLE, "The XLSX file is too large when unpacked.")
        try:
            shared: list[str] = []
            if "xl/sharedStrings.xml" in zf.namelist():
                root = _xml(zf.read("xl/sharedStrings.xml"))
                for si in root.findall("m:si", _NS):
                    shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
            date_styles = _date_styles(zf)
            sheet = _xml(zf.read(_first_sheet_path(zf)))
        except (KeyError, ET.ParseError):
            raise AppError(ErrorCode.FILE_UNREADABLE, "The XLSX file is damaged.") from None

    raw_rows: list[list[object]] = []
    for row in sheet.findall("m:sheetData/m:row", _NS):
        values: dict[int, object] = {}
        for c in row.findall("m:c", _NS):
            idx = _col_index(c.get("r", ""))
            kind = c.get("t")
            v = c.find("m:v", _NS)
            value: object = None
            if kind == "inlineStr":
                value = "".join(t.text or "" for t in c.iter(f"{{{_NS['m']}}}t"))
            elif v is not None and v.text is not None:
                if kind == "s":
                    value = shared[int(v.text)] if int(v.text) < len(shared) else None
                elif kind == "b":
                    value = "true" if v.text == "1" else "false"
                elif kind in ("str", "e"):
                    value = v.text
                else:
                    value = _number(v.text, int(c.get("s", "0")) in date_styles)
            values[idx] = value
        if values:
            width = max(values) + 1
            raw_rows.append([values.get(i) for i in range(width)])

    normalized = normalize_table(raw_rows)
    if not normalized:
        raise AppError(ErrorCode.NO_TABLE_FOUND_IN_FILE, "The first sheet has no header and data rows.")
    header, rows = normalized
    return Extraction(source="xlsx", pages=1,
                      tables=[ExtractedTable(header=header, rows=rows, method="xlsx")])


def _number(text: str, is_date: bool) -> str:
    if is_date:
        try:
            moment = _EXCEL_EPOCH + timedelta(days=float(text))
            return moment.date().isoformat() if moment.time() == datetime.min.time() \
                else moment.isoformat(sep=" ", timespec="seconds")
        except (ValueError, OverflowError):
            return text
    if re.fullmatch(r"-?\d+\.0+", text):
        return text.split(".")[0]
    return text
