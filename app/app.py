from fastapi import FastAPI

from app.ai.api.deps import current_user_id as ai_current_user_id
from app.ai.api.routes import router as ai_router
from app.ai.lifecycle import lifespan as ai_lifespan
from app.api.router import api_router
from app.core.config import settings
from app.security.auth import get_current_user_id


app = FastAPI(
    title=settings.app_name,
    description="Natural language interface for database queries",
    version=settings.app_version,
    lifespan=ai_lifespan,
)

app.include_router(api_router, prefix=settings.api_v1_prefix)
app.include_router(ai_router, prefix=settings.api_v1_prefix)

# Reuse the main app's authenticated user ID for all AI endpoints.
app.dependency_overrides[ai_current_user_id] = get_current_user_id


@app.get("/", tags=["Root"])
def root() -> dict[str, str]:
    return {
        "message": f"Welcome to {settings.app_name}",
        "docs": "/docs",
    }