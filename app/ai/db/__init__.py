"""AI <-> PostgreSQL bridge."""
from app.ai.db.engine import dispose_engines, get_read_engine, get_write_engine, ping
from app.ai.db.reader import QueryExecutionError, QueryResult, run_select
from app.ai.db.schema_inspector import (
    get_allowed_tables,
    get_schema,
    get_schema_context,
    get_table_names,
    invalidate_schema_cache,
)
from app.ai.db.sql_guard import SQLGuardError, validate_sql
from app.ai.db.writer import (
    get_target,
    insert_records,
    list_targets,
    register_target,
    update_record,
)

__all__ = [
    "dispose_engines", "get_read_engine", "get_write_engine", "ping",
    "run_select", "QueryResult", "QueryExecutionError",
    "get_schema", "get_schema_context", "get_table_names", "get_allowed_tables",
    "invalidate_schema_cache",
    "validate_sql", "SQLGuardError",
    "register_target", "get_target", "insert_records", "update_record", "list_targets",
]
