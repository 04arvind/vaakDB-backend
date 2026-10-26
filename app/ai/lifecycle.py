"""Startup / shutdown for the AI module.

    from fastapi import FastAPI
    from app.ai.api.routes import router as ai_router
    from app.ai.lifecycle import lifespan

    app = FastAPI(lifespan=lifespan)
    app.include_router(ai_router)

If the main app already has a lifespan, call `await startup()` and
`await shutdown()` from it instead.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

log = logging.getLogger(__name__)


async def startup() -> None:
    from app.ai.config import settings
    from app.ai.db.targets import register_targets
    from app.ai.utils.logger import configure_logging

    configure_logging()
    register_targets()

    if settings.AI_AUTO_CREATE_TABLES:
        from app.ai.memory.tables import ensure_memory_schema

        await ensure_memory_schema()
        log.info("AI memory tables ensured")

    from app.ai.providers.registry import configured_providers

    providers = configured_providers()
    if not providers:
        log.warning("No AI provider configured (set GROQ_API_KEY and/or GEMINI_API_KEY)")
    log.info("AI module started (env=%s, providers=%s)", settings.AI_ENV, providers)


async def shutdown() -> None:
    from app.ai.db.engine import dispose_engines
    from app.ai.db.vector_store import close_vector_store
    from app.ai.orchestration.orchestrator import orchestrator
    from app.ai.providers.registry import close_providers

    await orchestrator.shutdown()
    await close_providers()
    await dispose_engines()
    close_vector_store()
    log.info("AI module stopped")


@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    await startup()
    try:
        yield
    finally:
        await shutdown()
