"""Fake chat model with the same interface as app.ai.dbops.llm (structured output).

    fake = FakeChatModel({
        "IntentDecision": [{"intent": "ingest_file", "schema": "college", "table": "students"}],
        "IngestDraft": [draft_dict],
    })

Each schema name maps to a queue of responses: a dict (validated against the
requested Pydantic model, like the real thing), a callable(messages) -> dict,
or an Exception instance to raise. Every call is recorded in `fake.calls`.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from app.ai.dbops.llm import LLMBadOutputError


@dataclass
class Call:
    schema: str
    messages: list[tuple[str, str]]

    @property
    def system(self) -> str:
        return "\n".join(c for r, c in self.messages if r == "system")

    @property
    def user(self) -> str:
        return "\n".join(c for r, c in self.messages if r != "system")


class _Runnable:
    def __init__(self, fake: FakeChatModel, schema: type[BaseModel]):
        self.fake, self.schema = fake, schema

    async def ainvoke(self, messages: Sequence[tuple[str, str]]) -> Any:
        name = self.schema.__name__
        self.fake.calls.append(Call(name, list(messages)))
        queue = self.fake.responses.get(name)
        if not queue:
            raise AssertionError(f"FakeChatModel: no response queued for {name}")
        item = queue.pop(0) if len(queue) > 1 or self.fake.consume_last else queue[0]
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(messages)
        try:
            return self.schema.model_validate(item)
        except ValidationError as exc:  # the real wrapper re-asks, then gives up like this
            raise LLMBadOutputError(str(exc)) from exc


class FakeChatModel:
    def __init__(self, responses: dict[str, list[Any]] | None = None, *, consume_last: bool = False):
        self.responses: dict[str, list[Any]] = {k: list(v) for k, v in (responses or {}).items()}
        self.calls: list[Call] = []
        self.consume_last = consume_last

    def with_structured_output(self, schema: type[BaseModel]) -> _Runnable:
        return _Runnable(self, schema)

    def calls_for(self, schema: str) -> list[Call]:
        return [c for c in self.calls if c.schema == schema]


def deps_with(fake: FakeChatModel, *, run_select: Callable[..., Any] | None = None,
              record_history: bool = False) -> Any:
    from app.services.chat_service import ChatDeps

    deps = ChatDeps(chat_model_factory=lambda: fake, record_history=record_history)
    if run_select is not None:
        deps.run_select = run_select
    return deps


STUDENTS_DRAFT = {
    "schema": "college",
    "table": "students",
    "table_index": 0,
    "if_exists": "fail",
    "columns": [
        {"source_index": 0, "name": "roll_no", "pg_type": "integer", "primary_key": True,
         "nullable": False},
        {"source_index": 1, "name": "name", "pg_type": "varchar(100)"},
        {"source_index": 2, "name": "age", "pg_type": "integer"},
        {"source_index": 3, "name": "date_of_birth", "pg_type": "date"},
        {"source_index": 4, "name": "gpa", "pg_type": "numeric(4,2)"},
    ],
}
