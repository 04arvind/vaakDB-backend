"""Token estimates without a tokenizer dependency.

Groq (Llama) and Gemini use different tokenizers and neither ships a cheap
local one, so we estimate: characters / 3.5. That over-counts English slightly,
which is what we want for budgeting (never overflow the context window).
"""
from __future__ import annotations

import math
from typing import Iterable

CHARS_PER_TOKEN = 3.5
MESSAGE_OVERHEAD = 4  # rough per-message formatting cost
TRUNCATION_MARKER = "...[truncated]"


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN) if text else 0


def estimate_messages_tokens(contents: Iterable[str]) -> int:
    return sum(estimate_tokens(c) + MESSAGE_OVERHEAD for c in contents)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    keep = max(int(max_tokens * CHARS_PER_TOKEN) - len(TRUNCATION_MARKER), 0)
    return text[:keep] + TRUNCATION_MARKER
