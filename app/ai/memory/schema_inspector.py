"""Backward-compatible alias. The implementation lives in app.ai.db.schema_inspector."""
from app.ai.db.schema_inspector import (
    ColumnInfo,
    SchemaInfo,
    TableInfo,
    get_allowed_tables,
    get_hidden_tables,
    get_schema,
    get_schema_context,
    get_table_names,
    invalidate_schema_cache,
)

__all__ = [
    "ColumnInfo", "SchemaInfo", "TableInfo",
    "get_schema", "get_schema_context", "get_table_names",
    "get_allowed_tables", "get_hidden_tables", "invalidate_schema_cache",
]
