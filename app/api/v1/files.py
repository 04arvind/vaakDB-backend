from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, File, Path, UploadFile, status

from app.api.deps import CurrentCaller, DbSession
from app.core.exceptions import AppError, ErrorCode
from app.db.models import Upload
from app.schemas.result import UploadOut
from app.services.upload_service import handle_upload, upload_view

router = APIRouter(prefix="/files", tags=["Files"])


@router.post("/upload", status_code=status.HTTP_201_CREATED)
def upload_file(
    db: DbSession,
    caller: CurrentCaller,
    file: Annotated[UploadFile, File(description="PDF (text-based), CSV or XLSX, max MAX_UPLOAD_MB")],
) -> dict[str, Any]:
    """Store the file and extract its tables. Returns upload_id + extraction summary.

    The type is checked by magic bytes; the client's filename is only kept for display.
    """
    return handle_upload(db, caller, file.file, filename=file.filename,
                         content_type=file.content_type)


@router.get("/{upload_id}", response_model=UploadOut)
def get_upload(db: DbSession, caller: CurrentCaller,
               upload_id: Annotated[uuid.UUID, Path()]) -> dict[str, Any]:
    upload = db.get(Upload, upload_id)
    if upload is None or upload.user_id != caller.user_id:
        raise AppError(ErrorCode.UPLOAD_NOT_FOUND)
    return upload_view(upload)
