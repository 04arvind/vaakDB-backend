"""Typed application errors with stable codes.

Every error the API returns has the shape
    {"error": {"code": "TABLE_EXISTS", "message": "...", "details": {...}}}

Messages are written for end users. Stack traces, SQL text, driver errors and
API keys are never put in a response: unexpected exceptions become
INTERNAL_ERROR and the details only go to the server log.
"""
from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger(__name__)


class ErrorCode(StrEnum):
    # spec codes
    FORBIDDEN_SCHEMA = "FORBIDDEN_SCHEMA"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    SCANNED_PDF_UNSUPPORTED = "SCANNED_PDF_UNSUPPORTED"
    NO_TABLE_FOUND_IN_FILE = "NO_TABLE_FOUND_IN_FILE"
    TABLE_EXISTS = "TABLE_EXISTS"
    INVALID_IDENTIFIER = "INVALID_IDENTIFIER"
    TYPE_CAST_FAILED = "TYPE_CAST_FAILED"
    ROW_LIMIT_EXCEEDED = "ROW_LIMIT_EXCEEDED"
    OPERATION_EXPIRED = "OPERATION_EXPIRED"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    LLM_BAD_OUTPUT = "LLM_BAD_OUTPUT"
    # additional
    NOT_AUTHENTICATED = "NOT_AUTHENTICATED"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    EMAIL_TAKEN = "EMAIL_TAKEN"
    REGISTRATION_DISABLED = "REGISTRATION_DISABLED"
    NOT_FOUND = "NOT_FOUND"
    SCHEMA_NOT_FOUND = "SCHEMA_NOT_FOUND"
    TABLE_NOT_FOUND = "TABLE_NOT_FOUND"
    COLUMN_NOT_FOUND = "COLUMN_NOT_FOUND"
    UPLOAD_NOT_FOUND = "UPLOAD_NOT_FOUND"
    OPERATION_NOT_FOUND = "OPERATION_NOT_FOUND"
    OPERATION_NOT_PENDING = "OPERATION_NOT_PENDING"
    OPERATION_IN_PROGRESS = "OPERATION_IN_PROGRESS"
    DESTRUCTIVE_OPERATION_BLOCKED = "DESTRUCTIVE_OPERATION_BLOCKED"
    PLAN_INVALID = "PLAN_INVALID"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    FILE_UNREADABLE = "FILE_UNREADABLE"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    DATABASE_UNAVAILABLE = "DATABASE_UNAVAILABLE"
    QUERY_FAILED = "QUERY_FAILED"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    CONFLICT = "CONFLICT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


_STATUS: dict[ErrorCode, int] = {
    ErrorCode.FORBIDDEN_SCHEMA: status.HTTP_403_FORBIDDEN,
    ErrorCode.PERMISSION_DENIED: status.HTTP_403_FORBIDDEN,
    ErrorCode.DESTRUCTIVE_OPERATION_BLOCKED: status.HTTP_403_FORBIDDEN,
    ErrorCode.SCANNED_PDF_UNSUPPORTED: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ErrorCode.NO_TABLE_FOUND_IN_FILE: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ErrorCode.FILE_UNREADABLE: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ErrorCode.TABLE_EXISTS: status.HTTP_409_CONFLICT,
    ErrorCode.INVALID_IDENTIFIER: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ErrorCode.TYPE_CAST_FAILED: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ErrorCode.PLAN_INVALID: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ErrorCode.VALIDATION_ERROR: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ErrorCode.ROW_LIMIT_EXCEEDED: status.HTTP_413_CONTENT_TOO_LARGE,
    ErrorCode.FILE_TOO_LARGE: status.HTTP_413_CONTENT_TOO_LARGE,
    ErrorCode.UNSUPPORTED_FILE_TYPE: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    ErrorCode.OPERATION_EXPIRED: status.HTTP_410_GONE,
    ErrorCode.LLM_UNAVAILABLE: status.HTTP_503_SERVICE_UNAVAILABLE,
    ErrorCode.DATABASE_UNAVAILABLE: status.HTTP_503_SERVICE_UNAVAILABLE,
    ErrorCode.NOT_CONFIGURED: status.HTTP_503_SERVICE_UNAVAILABLE,
    ErrorCode.LLM_BAD_OUTPUT: status.HTTP_502_BAD_GATEWAY,
    ErrorCode.NOT_AUTHENTICATED: status.HTTP_401_UNAUTHORIZED,
    ErrorCode.INVALID_CREDENTIALS: status.HTTP_401_UNAUTHORIZED,
    ErrorCode.EMAIL_TAKEN: status.HTTP_409_CONFLICT,
    ErrorCode.REGISTRATION_DISABLED: status.HTTP_403_FORBIDDEN,
    ErrorCode.NOT_FOUND: status.HTTP_404_NOT_FOUND,
    ErrorCode.SCHEMA_NOT_FOUND: status.HTTP_404_NOT_FOUND,
    ErrorCode.TABLE_NOT_FOUND: status.HTTP_404_NOT_FOUND,
    ErrorCode.COLUMN_NOT_FOUND: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ErrorCode.UPLOAD_NOT_FOUND: status.HTTP_404_NOT_FOUND,
    ErrorCode.OPERATION_NOT_FOUND: status.HTTP_404_NOT_FOUND,
    ErrorCode.OPERATION_NOT_PENDING: status.HTTP_409_CONFLICT,
    ErrorCode.OPERATION_IN_PROGRESS: status.HTTP_409_CONFLICT,
    ErrorCode.CONFLICT: status.HTTP_409_CONFLICT,
    ErrorCode.EXECUTION_FAILED: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ErrorCode.QUERY_FAILED: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ErrorCode.INTERNAL_ERROR: status.HTTP_500_INTERNAL_SERVER_ERROR,
}

_DEFAULT_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.FORBIDDEN_SCHEMA: "You don't have access to that schema.",
    ErrorCode.PERMISSION_DENIED: "Your role is not allowed to do that.",
    ErrorCode.SCANNED_PDF_UNSUPPORTED: (
        "This PDF has no selectable text (it looks scanned). OCR is not supported yet; "
        "please upload a text-based PDF, CSV or XLSX."
    ),
    ErrorCode.NO_TABLE_FOUND_IN_FILE: "No table could be found in the uploaded file.",
    ErrorCode.TABLE_EXISTS: "That table already exists.",
    ErrorCode.INVALID_IDENTIFIER: "Invalid schema, table or column name.",
    ErrorCode.TYPE_CAST_FAILED: "Some values do not match their column types.",
    ErrorCode.ROW_LIMIT_EXCEEDED: "Too many rows for one operation.",
    ErrorCode.OPERATION_EXPIRED: "This operation has expired. Please ask again to get a new preview.",
    ErrorCode.LLM_UNAVAILABLE: "The AI service is unavailable right now. Please try again shortly.",
    ErrorCode.LLM_BAD_OUTPUT: "The AI service returned an unusable answer. Please rephrase and try again.",
    ErrorCode.NOT_AUTHENTICATED: "Not authenticated.",
    ErrorCode.INVALID_CREDENTIALS: "Invalid email or password.",
    ErrorCode.NOT_FOUND: "Not found.",
    ErrorCode.TABLE_NOT_FOUND: "Table not found.",
    ErrorCode.OPERATION_NOT_FOUND: "Operation not found.",
    ErrorCode.UPLOAD_NOT_FOUND: "Upload not found.",
    ErrorCode.EXECUTION_FAILED: "The operation failed and was rolled back. No data was changed.",
    ErrorCode.DATABASE_UNAVAILABLE: "The database is unavailable right now.",
    ErrorCode.INTERNAL_ERROR: "Internal server error.",
}


class AppError(Exception):
    """Raise anywhere; the API turns it into {"error": {code, message, details}}."""

    def __init__(
        self,
        code: ErrorCode,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        status_code: int | None = None,
    ):
        self.code = code
        self.message = message or _DEFAULT_MESSAGES.get(code, code.value.replace("_", " ").capitalize())
        self.details = details or {}
        self.status_code = status_code or _STATUS.get(code, status.HTTP_400_BAD_REQUEST)
        super().__init__(f"{code}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code.value, "message": self.message}
        if self.details:
            out["details"] = jsonable_encoder(self.details)
        return out


def error_body(code: ErrorCode | str, message: str, details: dict[str, Any] | None = None) -> dict:
    err: dict[str, Any] = {"code": str(code), "message": message}
    if details:
        err["details"] = jsonable_encoder(details)
    return {"error": err}


_HTTP_CODES = {
    401: ErrorCode.NOT_AUTHENTICATED,
    403: ErrorCode.PERMISSION_DENIED,
    404: ErrorCode.NOT_FOUND,
    409: ErrorCode.CONFLICT,
    413: ErrorCode.FILE_TOO_LARGE,
    415: ErrorCode.UNSUPPORTED_FILE_TYPE,
}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            log.error("AppError %s on %s", exc.code, request.url.path)
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
        return JSONResponse({"error": exc.to_dict()}, status_code=exc.status_code, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = [
            {"loc": [str(p) for p in e.get("loc", ())], "msg": str(e.get("msg", ""))[:200]}
            for e in exc.errors()[:20]
        ]
        return JSONResponse(
            error_body(ErrorCode.VALIDATION_ERROR, "The request is invalid.", {"errors": problems}),
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, ErrorCode.INTERNAL_ERROR if exc.status_code >= 500
                               else ErrorCode.VALIDATION_ERROR)
        message = exc.detail if isinstance(exc.detail, str) else code.value
        return JSONResponse(error_body(code, message), status_code=exc.status_code,
                            headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            error_body(ErrorCode.INTERNAL_ERROR, "Internal server error."),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
