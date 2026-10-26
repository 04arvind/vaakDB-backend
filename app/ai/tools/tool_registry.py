"""Tools an LLM may call (function calling), described with JSON Schema.

    registry = default_registry()
    specs = registry.openai_specs()                     # pass to the provider
    result = await registry.call("query_database", {"sql": "..."}, ctx)

Arguments are validated with the tool's Pydantic model before the handler runs,
and every call gets a ToolContext with the authenticated user, so handlers never
trust identity claims coming from the model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ValidationError

from app.ai.exceptions import AIError

log = logging.getLogger(__name__)


class ToolError(AIError):
    """A tool call failed. The message is safe to return to the model."""


@dataclass(frozen=True)
class ToolContext:
    user_id: str
    session_id: str | None = None
    dry_run: bool = False


Handler = Callable[[BaseModel, ToolContext], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Handler
    writes: bool = False  # tools that change data need explicit opt-in by the caller

    def spec(self) -> dict[str, Any]:
        """OpenAI/Groq-style function spec (Gemini accepts the same JSON Schema)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


@dataclass
class ToolRegistry:
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(f"Unknown tool: {name}") from None

    def names(self, *, include_writes: bool = False) -> list[str]:
        return sorted(n for n, t in self._tools.items() if include_writes or not t.writes)

    def openai_specs(self, *, include_writes: bool = False) -> list[dict[str, Any]]:
        return [self._tools[n].spec() for n in self.names(include_writes=include_writes)]

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        ctx: ToolContext,
        *,
        allow_writes: bool = False,
    ) -> dict[str, Any]:
        tool = self.get(name)
        if tool.writes and not allow_writes:
            raise ToolError(f"Tool '{name}' changes data and is not enabled for this call.")
        try:
            args = tool.args_model.model_validate(arguments or {})
        except ValidationError as exc:
            raise ToolError(f"Invalid arguments for {name}: {exc.errors()[:3]}") from exc
        log.info("Tool call: %s (writes=%s)", name, tool.writes)
        return await tool.handler(args, ctx)


_default: ToolRegistry | None = None


def default_registry() -> ToolRegistry:
    """Registry with the built-in database tools."""
    global _default
    if _default is None:
        from app.ai.tools.db_query_tool import QUERY_TOOL
        from app.ai.tools.db_write_tool import WRITE_TOOL

        registry = ToolRegistry()
        registry.register(QUERY_TOOL)
        registry.register(WRITE_TOOL)
        _default = registry
    return _default
