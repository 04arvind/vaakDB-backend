"""Unstructured text -> validated Pydantic objects ready for writer.py.

Pass the Pydantic model that matches your target table. The agent shows the
model's JSON schema to the LLM, validates the reply, and retries with the
validation error if the first attempt is malformed.
"""
from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.ai.agents.base import AgentError, BaseAgent
from app.ai.prompts.loader import load_prompt, render
from app.ai.utils.token_counter import truncate_to_tokens

T = TypeVar("T", bound=BaseModel)

EXTRACT_SYSTEM = load_prompt("extractor")

_ONE = "Return a single JSON object."
_MANY = 'Return {"items": [ ... ]} with one object per record found.'
MAX_INPUT_TOKENS = 12_000


class ExtractorAgent(BaseAgent):
    default_provider = "groq"
    temperature = 0.0
    max_tokens = 2000

    async def extract(
        self, raw_text: str, schema: type[T], *, hints: str = "", max_retries: int = 1
    ) -> T:
        return await self._run(raw_text, schema, False, hints, max_retries)

    async def extract_many(
        self, raw_text: str, schema: type[T], *, hints: str = "", max_retries: int = 1
    ) -> list[T]:
        return await self._run(raw_text, schema, True, hints, max_retries)

    async def _run(
        self,
        raw_text: str,
        schema: type[T],
        many: bool,
        hints: str,
        max_retries: int,
    ) -> Any:
        if not raw_text or not raw_text.strip():
            raise AgentError("Nothing to extract: the text is empty.")

        system = render(
            EXTRACT_SYSTEM,
            SCHEMA=json.dumps(schema.model_json_schema(), indent=2),
            SHAPE=_MANY if many else _ONE,
        )
        text = truncate_to_tokens(raw_text, MAX_INPUT_TOKENS)
        base_prompt = (f"Hints: {hints}\n\n" if hints else "") + f"Text:\n<<<\n{text}\n>>>"
        last_error: str | None = None

        for _ in range(max_retries + 1):
            prompt = base_prompt
            if last_error:
                prompt += (
                    f"\n\nYour previous answer was invalid: {last_error}\n"
                    "Return corrected JSON only."
                )
            resp = await self._ask(system, prompt, json_mode=True)

            try:
                data = self.parse_json(resp.text)
                if many:
                    if isinstance(data, dict):
                        items = data["items"] if "items" in data else [data]
                    else:
                        items = data
                    if not isinstance(items, list):
                        raise ValueError('"items" must be a list.')
                    return [schema.model_validate(item) for item in items]
                if isinstance(data, list) and len(data) == 1:
                    data = data[0]
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                last_error = str(exc)[:500]

        raise AgentError(f"Extraction failed after retries: {last_error}")
