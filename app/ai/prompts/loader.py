"""Loads system prompts from prompts/system/*.md.

    SQL_SYSTEM = load_prompt("sql_agent")
    text = render(SQL_SYSTEM, TODAY="2026-01-01", SCHEMA=schema_text)

Placeholders are written <<NAME>> (not {name}), so JSON examples inside the
prompts need no escaping. render() fails loudly on a placeholder left unfilled.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "system"
_NAME = re.compile(r"^[a-z0-9_]+$")
_PLACEHOLDER = re.compile(r"<<([A-Z_]+)>>")


class PromptNotFoundError(FileNotFoundError):
    pass


@lru_cache(maxsize=64)
def load_prompt(name: str) -> str:
    if not _NAME.match(name):
        raise ValueError(f"Invalid prompt name: {name!r}")
    path = PROMPT_DIR / f"{name}.md"
    if not path.is_file():
        raise PromptNotFoundError(f"Prompt not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise PromptNotFoundError(f"Prompt file is empty: {path}")
    return text


def render(template: str, **values: object) -> str:
    """Fill <<KEY>> placeholders in one pass (values are never re-scanned)."""

    def sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"Missing prompt value: {key}")
        return str(values[key])

    return _PLACEHOLDER.sub(sub, template)
