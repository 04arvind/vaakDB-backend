"""Routing decisions: which workflow handles a chat message, and which provider
handles a task.

route_message() uses cheap rules first (greetings, explicit table names) and
asks a fast LLM only when the rules are unsure. If the LLM call fails, the
keyword heuristic decides, so routing never fails a request.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from app.ai.agents.base import BaseAgent
from app.ai.prompts.loader import load_prompt, render

log = logging.getLogger(__name__)

Route = Literal["general", "data_query", "knowledge"]
Task = Literal["chat", "sql", "extract", "summarize", "memory", "file"]

ROUTER_SYSTEM = load_prompt("router")
MAX_TABLES_IN_PROMPT = 100

_GREETING = re.compile(
    r"^\s*(hi|hii+|hello|hey|yo|thanks|thank you|thx|ok|okay|bye|good\s+(morning|afternoon|"
    r"evening|night)|who are you|what can you do|help)\b[\s!.?]*$",
    re.IGNORECASE,
)
_DATA_HINT = re.compile(
    r"\b(how many|how much|count|total|sum|average|avg|mean|median|max(imum)?|min(imum)?|"
    r"list|show|top \d+|bottom \d+|last (week|month|quarter|year)|this (week|month|quarter|year)|"
    r"yesterday|today|between|per (day|week|month)|group(ed)? by|trend|compare|"
    r"revenue|sales|orders?|customers?|users?|records?|rows?|table)\b",
    re.IGNORECASE,
)
_KNOWLEDGE_HINT = re.compile(
    r"\b(documents?|docs?|files?|pdfs?|upload(ed|s)?|attachments?|reports?|polic(y|ies)|"
    r"contracts?|invoices?|receipts?|notes?|manuals?|guides?|according to|mentioned in)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z0-9]+")
GREETING_REASON = "greeting/small talk"


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    confidence: float
    reason: str


class _RouterAgent(BaseAgent):
    default_provider = "groq"
    temperature = 0.0
    max_tokens = 60


def _mentions_table(message: str, tables: list[str]) -> bool:
    words = set(_WORD.findall(message.lower()))
    for table in tables:
        name = table.lower()
        parts = set(name.split("_"))
        singular = name[:-1] if name.endswith("s") else name
        if name in words or singular in words or (len(parts) > 1 and parts <= words):
            return True
    return False


def heuristic_route(message: str, tables: list[str]) -> RouteDecision:
    if _GREETING.match(message):
        return RouteDecision("general", 0.95, GREETING_REASON)
    if _KNOWLEDGE_HINT.search(message):
        return RouteDecision("knowledge", 0.75, "document keywords")
    if not tables:
        return RouteDecision("general", 0.6, "no tables available")
    if _mentions_table(message, tables):
        return RouteDecision("data_query", 0.9, "mentions a table")
    if _DATA_HINT.search(message):
        return RouteDecision("data_query", 0.6, "data keywords")
    return RouteDecision("general", 0.5, "no data signal")


async def route_message(
    message: str,
    tables: list[str],
    *,
    provider: str | None = None,
    use_llm: bool = True,
) -> RouteDecision:
    rule = heuristic_route(message, tables)
    if rule.confidence >= 0.9 or not use_llm:
        return rule

    try:
        agent = _RouterAgent(provider=provider)
        system = render(
            ROUTER_SYSTEM, TABLES=", ".join(tables[:MAX_TABLES_IN_PROMPT]) or "(none)"
        )
        resp = await agent._ask(system, message, json_mode=True)
        data = agent.parse_json(resp.text)
        route = data.get("route")
        if route not in ("general", "data_query", "knowledge"):
            raise ValueError(f"bad route {route!r}")
        confidence = float(data.get("confidence", 0.7))
        if route == "data_query" and not tables:
            route = "knowledge"  # nothing to query; documents may still answer it
        return RouteDecision(route, max(0.0, min(confidence, 1.0)), "llm")
    except Exception as exc:  # routing must never fail the request
        log.warning("LLM routing failed (%s); using heuristic", exc)
        return rule


def choose_provider(
    task: Task,
    *,
    text_len: int = 0,
    has_attachments: bool = False,
    override: str | None = None,
) -> str:
    """Preferred provider for a task. Unconfigured providers are skipped later by
    orchestration/fallback.py, so this only expresses preference."""
    from app.ai.config import settings

    if override:
        return override
    if has_attachments or task == "file":
        return "gemini"
    if text_len > settings.LONG_CONTEXT_CHARS or task == "summarize":
        return "gemini"
    return "groq"
