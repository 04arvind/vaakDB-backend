"""Uploads: store the file, extract its tables, record and audit it."""
from __future__ import annotations

import uuid
from typing import Any, BinaryIO

from sqlalchemy.orm import Session

from app.core.exceptions import AppError, ErrorCode
from app.db.models import Upload
from app.services.audit_service import audit
from app.services.ingest import Extraction, get_cached_extraction
from app.services.ingest.storage import (
    relative_storage_path,
    resolve_stored_path,
    sha256_file,
    store_upload,
)
from app.services.metadata_ops import Caller


def handle_upload(db: Session, caller: Caller, stream: BinaryIO, *, filename: str | None,
                  content_type: str | None) -> dict[str, Any]:
    stored = store_upload(stream, user_id=caller.user_id, client_name=filename,
                          client_type=content_type)
    upload = Upload(
        id=uuid.uuid4(),
        user_id=caller.user_id,
        filename=stored.filename,
        mime=stored.mime,
        size_bytes=stored.size_bytes,
        sha256=stored.sha256,
        storage_path=relative_storage_path(stored.path),
        status="stored",
    )
    db.add(upload)
    db.commit()

    try:
        extraction = get_cached_extraction(stored.sha256, stored.path, stored.mime)
        if not extraction.tables:
            raise AppError(ErrorCode.NO_TABLE_FOUND_IN_FILE,
                           "No table could be found in this file. Make sure it contains a "
                           "table with a header row.")
    except AppError as exc:
        upload.status = "unsupported" if exc.code in (
            ErrorCode.SCANNED_PDF_UNSUPPORTED, ErrorCode.NO_TABLE_FOUND_IN_FILE) else "failed"
        upload.extraction = {"error": exc.to_dict()}
        db.commit()
        audit("upload", user_id=caller.user_id, status=upload.status, upload_id=upload.id,
              error=f"{exc.code}: {exc.message}",
              details={"mime": stored.mime, "size_bytes": stored.size_bytes})
        exc.details = {**exc.details, "upload_id": str(upload.id)}
        raise

    summary = extraction.summary()
    upload.status = "extracted"
    upload.extraction = summary
    db.commit()
    audit("upload", user_id=caller.user_id, status="extracted", upload_id=upload.id,
          row_count=sum(len(t.rows) for t in extraction.tables),
          details={"mime": stored.mime, "size_bytes": stored.size_bytes,
                   "sha256": stored.sha256, "tables": len(extraction.tables)})
    return {
        "upload_id": str(upload.id),
        "filename": upload.filename,
        "mime": upload.mime,
        "size_bytes": upload.size_bytes,
        "sha256": upload.sha256,
        "status": upload.status,
        "extraction": summary,
    }


def load_upload(db: Session, caller: Caller, upload_id: uuid.UUID | str) -> tuple[Upload, Extraction]:
    """Owner-only. The file is re-hashed so a swapped file on disk is detected."""
    try:
        uid = upload_id if isinstance(upload_id, uuid.UUID) else uuid.UUID(str(upload_id))
    except ValueError:
        raise AppError(ErrorCode.UPLOAD_NOT_FOUND) from None
    upload = db.get(Upload, uid)
    if upload is None or upload.user_id != caller.user_id:
        raise AppError(ErrorCode.UPLOAD_NOT_FOUND)
    if upload.status != "extracted":
        err = (upload.extraction or {}).get("error") or {}
        code = err.get("code")
        if isinstance(code, str) and code in ErrorCode.__members__:
            raise AppError(ErrorCode(code), err.get("message"))
        raise AppError(ErrorCode.FILE_UNREADABLE, "This upload could not be read.")
    path = resolve_stored_path(upload.storage_path)
    if not path.is_file() or sha256_file(path) != upload.sha256:
        raise AppError(ErrorCode.UPLOAD_NOT_FOUND, "The uploaded file is no longer available.")
    return upload, get_cached_extraction(upload.sha256, path, upload.mime)


def upload_view(upload: Upload) -> dict[str, Any]:
    return {
        "upload_id": str(upload.id),
        "filename": upload.filename,
        "mime": upload.mime,
        "size_bytes": upload.size_bytes,
        "sha256": upload.sha256,
        "status": upload.status,
        "created_at": upload.created_at.isoformat() if upload.created_at else None,
        "extraction": upload.extraction,
    }
