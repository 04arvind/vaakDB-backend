import io
import zipfile
from pathlib import Path

import pytest

from app.core.exceptions import AppError, ErrorCode
from app.services.ingest import extract_file
from app.services.ingest.pdf_extractor import _text_blocks, extract_pdf
from app.services.ingest.storage import CSV, PDF, XLSX, detect_mime, sanitize_filename
from app.services.ingest.tabular import extract_csv, extract_xlsx

FIX = Path(__file__).resolve().parents[1] / "fixtures"


def test_students_pdf_table() -> None:
    ex = extract_pdf(FIX / "students.pdf")
    assert ex.pages == 1 and len(ex.tables) == 1
    t = ex.tables[0]
    assert t.header == ["Roll No", "Name", "Age", "Date of Birth", "GPA"]
    assert len(t.rows) == 5 and t.rows[0] == ["101", "Asha Verma", "20", "2004-05-14", "8.70"]


def test_scanned_pdf() -> None:
    with pytest.raises(AppError) as exc:
        extract_pdf(FIX / "scanned.pdf")
    assert exc.value.code == ErrorCode.SCANNED_PDF_UNSUPPORTED


def test_no_tables_pdf() -> None:
    ex = extract_pdf(FIX / "no_tables.pdf")
    assert ex.tables == [] and "student council" in ex.raw_text_excerpt


def test_text_block_fallback() -> None:
    text = "Results\nName    Score    Grade\nAsha    91    A\nRahul    78    B\nend of list"
    tables = _text_blocks([(1, text)])
    assert len(tables) == 1 and tables[0].header == ["Name", "Score", "Grade"]
    assert tables[0].rows == [["Asha", "91", "A"], ["Rahul", "78", "B"]]


def test_corrupt_pdf(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4\nthis is not really a pdf")
    with pytest.raises(AppError) as exc:
        extract_file(bad, PDF)
    assert exc.value.code in (ErrorCode.FILE_UNREADABLE, ErrorCode.SCANNED_PDF_UNSUPPORTED)


def test_csv(tmp_path: Path) -> None:
    p = tmp_path / "s.csv"
    p.write_text("Roll No;Name;Age\n1;Asha;20\n2;Rahul;\n\n", encoding="utf-8")
    t = extract_csv(p).tables[0]
    assert t.header == ["Roll No", "Name", "Age"] and t.rows == [["1", "Asha", "20"], ["2", "Rahul", None]]


def _xlsx(path: Path, *, extra_xml: str = "") -> None:
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    rel_ns = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("xl/workbook.xml", f'<workbook {ns} {rel_ns}><sheets><sheet name="S" sheetId="1" '
                                       f'r:id="rId1"/></sheets></workbook>')
        zf.writestr("xl/_rels/workbook.xml.rels",
                    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                    '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        zf.writestr("xl/sharedStrings.xml", f"{extra_xml}<sst {ns}><si><t>Name</t></si><si><t>Joined</t>"
                                            f"</si><si><t>Asha</t></si></sst>")
        zf.writestr("xl/styles.xml", f'<styleSheet {ns}><cellXfs><xf numFmtId="0"/><xf numFmtId="14"/>'
                                     f"</cellXfs></styleSheet>")
        zf.writestr("xl/worksheets/sheet1.xml",
                    f'<worksheet {ns}><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c>'
                    f'<c r="B1" t="s"><v>1</v></c><c r="C1" t="inlineStr"><is><t>Score</t></is></c></row>'
                    f'<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2" s="1"><v>45306</v></c>'
                    f'<c r="C2"><v>91.0</v></c></row></sheetData></worksheet>')


def test_xlsx(tmp_path: Path) -> None:
    p = tmp_path / "s.xlsx"
    _xlsx(p)
    t = extract_xlsx(p).tables[0]
    assert t.header == ["Name", "Joined", "Score"]
    assert t.rows == [["Asha", "2024-01-15", "91"]]


def test_xlsx_rejects_entities(tmp_path: Path) -> None:
    p = tmp_path / "bomb.xlsx"
    _xlsx(p, extra_xml='<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]>')
    with pytest.raises(AppError) as exc:
        extract_xlsx(p)
    assert exc.value.code == ErrorCode.FILE_UNREADABLE


def test_detect_mime_by_magic_bytes(tmp_path: Path) -> None:
    pdf = (FIX / "students.pdf").read_bytes()
    p = tmp_path / "x"
    p.write_bytes(pdf)
    assert detect_mime(p, pdf[:100], "evil.csv", "text/csv") == PDF  # content wins over name
    xl = tmp_path / "y.xlsx"
    _xlsx(xl)
    assert detect_mime(xl, xl.read_bytes()[:100], "y.bin", None) == XLSX
    assert detect_mime(p, b"a,b\n1,2\n", "data.csv", None) == CSV
    for head, name in ((b"MZ\x90\x00binary", "x.csv"), (b"a,b\n1,2", "x.exe"), (b"PK\x03\x04", "x.zip")):
        with pytest.raises(AppError) as exc:
            detect_mime(p if head.startswith(b"PK") is False else _zip(tmp_path), head, name, None)
        assert exc.value.code == ErrorCode.UNSUPPORTED_FILE_TYPE


def _zip(tmp_path: Path) -> Path:
    z = tmp_path / "plain.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "hi")
    z.write_bytes(buf.getvalue())
    return z


def test_sanitize_filename() -> None:
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("C:\\x\\a\x00b.pdf") == "ab.pdf"
    assert sanitize_filename(None) == "upload"
