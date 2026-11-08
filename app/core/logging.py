"""Logging setup for the backend. Secrets are scrubbed from every record."""
from __future__ import annotations

import logging
import re

_SECRET_PATTERNS = (
    # password inside a connection URL
    (re.compile(r"(://[^:/@\s]+:)[^\s]+@"), r"\1***@"),
    # key=value / "key": "value" style secrets
    (re.compile(r"(?i)((?:api[_-]?key|password|secret|token)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]+"),
     r"\1***"),
    # well-known API key shapes (Groq gsk_..., Google AIza...)
    (re.compile(r"\b(gsk_|AIza)[A-Za-z0-9_\-]{8,}"), r"\1***"),
)


def scrub(text: str) -> str:
    for pattern, repl in _SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    return text


class SecretScrubber(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        cleaned = scrub(message)
        if cleaned != message:
            record.msg, record.args = cleaned, None
        return True


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if not any(isinstance(f, SecretScrubber) for h in root.handlers for f in h.filters):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-5s [%(name)s] %(message)s")
        )
        handler.addFilter(SecretScrubber())
        root.addHandler(handler)
    root.setLevel(level.upper())
    logging.getLogger("pdfminer").setLevel(logging.WARNING)
