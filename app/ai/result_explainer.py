"""Backward-compatible alias. Result explanation is SQLAgent.answer() (short answers)
and SummarizerAgent.summarize_rows() (longer summaries of query results)."""
from app.ai.agents.sql_agent import SQLAgent  # noqa: F401
from app.ai.agents.summarizer_agent import SummarizerAgent  # noqa: F401
