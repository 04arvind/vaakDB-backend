"""Natural language -> PostgreSQL SELECT, plus result explanation.

This agent only GENERATES sql. Validation (sql_guard) and execution (reader)
happen in the workflow, never here.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any

from pydantic import BaseModel

from app.ai.agents.base import AgentError, BaseAgent
from app.ai.prompts.loader import load_prompt, render
from app.ai.schemas.llm_io import Message
from app.ai.utils.token_counter import truncate_to_tokens


class CannotAnswer(AgentError):
    """The question cannot be answered from the available schema."""


class SQLResult(BaseModel):
    sql: str
    explanation: str = ""


SQL_SYSTEM = load_prompt("sql_agent")
ANSWER_SYSTEM = load_prompt("answer")
MAX_RESULT_TOKENS = 6000


class SQLAgent(BaseAgent):
    default_provider = "groq"
    temperature = 0.0
    max_tokens = 800

    def __init__(self, *args: Any, max_rows: int = 100, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.max_rows = max_rows

    def _system(self, schema: str) -> str:
        return render(
            SQL_SYSTEM,
            TODAY=date.today().isoformat(),
            MAX_ROWS=self.max_rows,
            SCHEMA=schema or "(no tables available)",
        )

    async def generate_sql(
        self,
        question: str,
        schema: str,
        history: list[Message] | None = None,
    ) -> SQLResult:
        resp = await self._ask(
            self._system(schema), question, history=history, json_mode=True
        )
        return self._to_result(resp.text)

    async def repair_sql(
        self,
        question: str,
        schema: str,
        bad_sql: str,
        error: str,
    ) -> SQLResult:
        """Called by the workflow when the guard rejects the SQL or Postgres errors."""
        prompt = (
            f"Question: {question}\n\n"
            f"Your previous query:\n{bad_sql}\n\n"
            f"It failed with:\n{error}\n\n"
            "Return a corrected query as JSON."
        )
        resp = await self._ask(self._system(schema), prompt, json_mode=True)
        return self._to_result(resp.text)

    async def answer(
        self,
        question: str,
        sql: str,
        rows: list[dict[str, Any]],
        max_rows: int = 50,
        *,
        truncated: bool = False,
    ) -> str:
        shown = rows[:max_rows]
        note = ""
        if len(rows) > max_rows:
            note = f"\n(Showing {max_rows} of {len(rows)} rows.)"
        if truncated:
            note += "\n(The query returned more rows than the limit; only these were fetched.)"
        result_json = truncate_to_tokens(json.dumps(shown, default=str), MAX_RESULT_TOKENS)
        prompt = (
            f"Question: {question}\n\nSQL used:\n{sql}\n\n"
            f"Result ({len(rows)} rows):\n{result_json}{note}"
        )
        resp = await self._ask(ANSWER_SYSTEM, prompt, temperature=0.2, max_tokens=600)
        return resp.text.strip()

    def _to_result(self, raw: str) -> SQLResult:
        try:
            data = self.parse_json(raw)
            result = SQLResult.model_validate(data)
        except Exception as exc:  # malformed JSON or wrong shape
            raise AgentError(f"SQL agent returned invalid output: {exc}") from exc

        result.sql = result.sql.strip().rstrip(";").strip()
        if not result.sql:
            raise CannotAnswer(result.explanation or "Question not answerable from schema.")
        return result
