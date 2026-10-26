"""Logging for the AI module: stdlib logging plus a per-request id.

    from app.ai.utils.logger import configure_logging, request_id_var
    configure_logging()                       # once, at startup
    token = request_id_var.set("abc123")      # done per request by api/routes.py

Every record gets a `request_id` attribute, so log lines from one request can be
correlated across agents, providers and the database layer.
"""
from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

request_id_var: ContextVar[str] = ContextVar("ai_request_id", default="-")

_LOGGER_NAME = "app.ai"
_configured = False


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for log shippers (Loki, CloudWatch, ...)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None, *, json_logs: bool | None = None) -> None:
    """Attach a handler to the `app.ai` logger. Safe to call more than once."""
    global _configured
    from app.ai.config import settings

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel((level or settings.AI_LOG_LEVEL).upper())
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    use_json = settings.AI_LOG_JSON if json_logs is None else json_logs
    handler.setFormatter(
        JsonFormatter()
        if use_json
        else logging.Formatter(
            "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not any(isinstance(f, RequestIdFilter) for f in logger.filters):
        logger.addFilter(RequestIdFilter())
    return logger
