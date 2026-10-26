"""Backward-compatible alias. SQL generation lives in app.ai.agents.sql_agent and
the full question -> answer pipeline in app.ai.workflows.qa_workflow."""
from app.ai.agents.sql_agent import CannotAnswer, SQLAgent, SQLResult  # noqa: F401
from app.ai.workflows.qa_workflow import QAResult, run_qa  # noqa: F401
