"""Intent routing: message -> IntentDecision.

Clear metadata requests ("show tables in schema college", "describe
college.students") are matched by rules, without an LLM call. Everything else
goes to the LLM with structured output. Only the user's message (and the fact
that a file is attached) is shown to the router, never the file's content, so
a file cannot steer the intent or the target table.
"""
from __future__ import annotations

import re

from app.ai.dbops.llm import ChatModel
from app.ai.dbops.prompts import ROUTER_SYSTEM, render
from app.schemas.plans import Intent, IntentDecision

_NAME = r"[`\"']?([A-Za-z_][A-Za-z0-9_]*)[`\"']?"
_LIST_SCHEMAS = re.compile(
    r"^\s*(?:show|list|what are|which)\b(?:\s+(?:me|all|the|my|available))*\s+schemas?\b",
    re.IGNORECASE,
)
_LIST_TABLES = re.compile(
    rf"^\s*(?:show|list|what are|which)\b(?:\s+(?:me|all|the|available))*\s+tables?\s+"
    rf"(?:are\s+)?(?:in|of|from|inside)\s+(?:the\s+)?(?:schema\s+)?{_NAME}(?:\s+schema)?\s*\??\s*$",
    re.IGNORECASE,
)
_DESCRIBE = re.compile(
    rf"^\s*(?:describe|desc|explain the structure of|show (?:me )?(?:the )?(?:columns|structure|schema) "
    rf"(?:of|for))\s+(?:table\s+)?{_NAME}\.{_NAME}\s*\??\s*$",
    re.IGNORECASE,
)
_COLUMNS_OF = re.compile(
    rf"^\s*what\s+columns\s+(?:does|do|are in|are there in)\s+(?:table\s+)?{_NAME}\.{_NAME}"
    rf"(?:\s+have)?\s*\??\s*$",
    re.IGNORECASE,
)


def rule_based_intent(message: str) -> IntentDecision | None:
    text = message.strip()
    if _LIST_SCHEMAS.match(text):
        return IntentDecision(intent=Intent.list_schemas, reason="rule")
    m = _LIST_TABLES.match(text)
    if m:
        return IntentDecision(intent=Intent.list_tables, schema=m.group(1), reason="rule")
    m = _DESCRIBE.match(text) or _COLUMNS_OF.match(text)
    if m:
        return IntentDecision(intent=Intent.describe_table, schema=m.group(1), table=m.group(2),
                              reason="rule")
    return None


async def route_intent(
    message: str,
    *,
    chat_model: ChatModel,
    has_upload: bool,
    upload_filename: str | None = None,
    schemas: list[str] | None = None,
) -> IntentDecision:
    ruled = rule_based_intent(message)
    if ruled is not None:
        return ruled
    system = render(ROUTER_SYSTEM, SCHEMAS=", ".join(schemas or []) or "(none)")
    attached = (f"A file is attached ({upload_filename or 'upload'})." if has_upload
                else "No file is attached.")
    user = f"{attached}\n\nUser message:\n{message.strip()[:4000]}"
    return await chat_model.with_structured_output(IntentDecision).ainvoke(
        [("system", system), ("user", user)]
    )
