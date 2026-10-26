"""Backward-compatible alias. The implementation lives in app.ai.db.reader."""
from app.ai.db.reader import QueryExecutionError, QueryResult, run_select

__all__ = ["QueryExecutionError", "QueryResult", "run_select"]
