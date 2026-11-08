"""Upload storage.

- size capped while streaming (MAX_UPLOAD_MB)
- type decided by the file's magic bytes, never by the client's filename or
  Content-Type (CSV, which has no magic bytes, must also be valid text)
- stored under UPLOAD_DIR/<user_id>/<random>.<ext>; the client filename is only
  kept, sanitized, for display
- sha256 recorded, and re-checked whenever the file is read back
"""
from __future__ import annotations

import hashlib
import os
import re
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from app.core.config import settings
from app.core.exceptions import AppError, ErrorCode

PDF = "application/pdf"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
CSV = "text/csv"
EXTENSIONS = {PDF: "pdf", XLSX: "xlsx", CSV: "csv"}
_CHUNK = 1024 * 1024


@dataclass
class StoredFile:
    path: Path
    mime: str
    size_bytes: int
    sha256: str
    filename: str


def sanitize_filename(name: str | None) -> str:
    base = os.path.basename((name or "").replace("\\", "/"))
    base = re.sub(r"[\x00-\x1f\x7f]", "", base).strip()
    return base[:255] or "upload"


def _looks_like_text(head: bytes) -> bool:
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:
        if exc.start < len(head) - 4:  # not just a multi-byte char cut at the end
            try:
                head.decode("cp1252")
            except UnicodeDecodeError:
                return False
    printable = sum(1 for b in head if b >= 32 or b in (9, 10, 13))
    return printable / max(len(head), 1) > 0.97


def detect_mime(path: Path, head: bytes, client_name: str, client_type: str | None) -> str:
    if head.startswith(b"%PDF-"):
        return PDF
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as zf:
                if "xl/workbook.xml" in zf.namelist():
                    return XLSX
        except zipfile.BadZipFile:
            pass
        raise AppError(ErrorCode.UNSUPPORTED_FILE_TYPE, "Only PDF, CSV and XLSX files are supported.")
    declared_csv = client_name.lower().endswith((".csv", ".tsv", ".txt")) or (
        (client_type or "").split(";")[0].strip().lower() in ("text/csv", "text/plain",
                                                              "application/csv")
    )
    if declared_csv and _looks_like_text(head):
        return CSV
    raise AppError(ErrorCode.UNSUPPORTED_FILE_TYPE, "Only PDF, CSV and XLSX files are supported.")


def store_upload(stream: BinaryIO, *, user_id: int, client_name: str | None,
                 client_type: str | None) -> StoredFile:
    """Stream to disk with a size cap, hash, then verify the type by magic bytes."""
    folder = settings.upload_path / str(int(user_id))
    folder.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    tmp = folder / f"{token}.part"
    digest = hashlib.sha256()
    size = 0
    head = b""
    try:
        with open(tmp, "xb") as out:
            while chunk := stream.read(_CHUNK):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise AppError(ErrorCode.FILE_TOO_LARGE,
                                   f"Files are limited to {settings.max_upload_mb} MB.")
                if len(head) < 64 * 1024:
                    head += chunk[: 64 * 1024 - len(head)]
                digest.update(chunk)
                out.write(chunk)
        if size == 0:
            raise AppError(ErrorCode.FILE_UNREADABLE, "The uploaded file is empty.")
        filename = sanitize_filename(client_name)
        mime = detect_mime(tmp, head, filename, client_type)
        final = folder / f"{token}.{EXTENSIONS[mime]}"
        os.replace(tmp, final)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return StoredFile(path=final, mime=mime, size_bytes=size, sha256=digest.hexdigest(),
                      filename=filename)


def resolve_stored_path(storage_path: str) -> Path:
    """Stored paths are relative to UPLOAD_DIR; refuse anything that escapes it."""
    root = settings.upload_path.resolve()
    path = (root / storage_path).resolve()
    if root not in path.parents:
        raise AppError(ErrorCode.UPLOAD_NOT_FOUND)
    return path


def relative_storage_path(path: Path) -> str:
    return path.resolve().relative_to(settings.upload_path.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
