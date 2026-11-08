from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.core.exceptions import install_error_handlers
from app.core.logging import configure_logging

log = logging.getLogger(__name__)

_ai_enabled = False


def _mount_ai_routes(app: FastAPI) -> None:
    """The original AI module routes (/api/v1/ai/...) query one shared schema and
    are not isolated per user, so they are only mounted with ENABLE_AI_ROUTES=true."""
    global _ai_enabled
    if not settings.enable_ai_routes:
        return
    try:
        from app.ai.api.routes import router as ai_router

        app.include_router(ai_router, prefix=settings.api_v1_prefix)
        _ai_enabled = True
    except Exception as exc:  # pragma: no cover - depends on installed extras
        log.warning("AI routes not mounted (%s: %s)", type(exc).__name__, exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging("DEBUG" if settings.debug else "INFO")
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    if _ai_enabled:
        from app.ai.lifecycle import startup

        try:
            await startup()
        except Exception as exc:
            log.warning("AI module startup failed: %s", type(exc).__name__)
    try:
        yield
    finally:
        from app.db.target_db import dispose_engines

        dispose_engines()
        if _ai_enabled:
            from app.ai.lifecycle import shutdown

            try:
                await shutdown()
            except Exception:
                log.warning("AI module shutdown failed", exc_info=True)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="Natural language interface for database queries",
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "Accept", "X-Request-ID"],
    )
    install_error_handlers(app)
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    _mount_ai_routes(app)

    @app.get("/", tags=["Root"])
    def root() -> dict[str, str]:
        return {"message": f"Welcome to {settings.app_name}", "docs": "/docs"}

    return app


app = create_app()
