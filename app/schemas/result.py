"""The single response envelope used by /chat (and the operation endpoints).

    {"type": "...", "data": {...}, "message": "short natural-language summary"}

type                data
------------------  ----------------------------------------------------------------
schema_list         {columns, items: [{schema, table_count}]}
table_list          {schema, columns, items: [{table, kind, estimated_rows}]}
table_description   {schema, table, columns: [{name, type, nullable, default,
                     primary_key, references}], primary_key, row_count, sample_rows}
query_result        {columns, rows, sql?, truncated?, page?, total_rows?}
operation_preview   {operation_id, expires_at, operation_type, target, mode, ddl,
                     columns, column_mapping, sample_rows (first 10), total_rows,
                     rows_to_write, failed_rows, failed_row_count, affected_rows,
                     destructive, warnings}
operation_result    {operation_id, status, target, result: {rows_written | rows_updated
                     | rows_deleted, rows_skipped, ...}}
message             null or {clarification: true}
error               {code, message, details?}
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

ResponseType = Literal[
    "schema_list", "table_list", "table_description", "query_result",
    "operation_preview", "operation_result", "message", "error",
]


class Envelope(BaseModel):
    type: ResponseType
    data: Any = None
    message: str = ""
    conversation_id: str | None = None
    intent: str | None = None


class OperationOut(BaseModel):
    operation_id: str
    type: str
    status: str
    target: dict[str, str]
    preview: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str | None = None
    expires_at: str
    confirmed_at: str | None = None
    executed_at: str | None = None


class UploadOut(BaseModel):
    upload_id: str
    filename: str
    mime: str
    size_bytes: int
    sha256: str
    status: str
    created_at: str | None = None
    extraction: dict[str, Any] | None = None
