from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.chat import router as chat_router
from app.api.v1.connections import router as connections_router
from app.api.v1.files import router as files_router
from app.api.v1.health import router as health_router
from app.api.v1.operations import router as operations_router
from app.api.v1.projects import router as projects_router
from app.api.v1.queries import router as queries_router
from app.api.v1.schema import router as schema_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(auth_router)
api_router.include_router(files_router)
api_router.include_router(chat_router)
api_router.include_router(operations_router)
api_router.include_router(schema_router)
api_router.include_router(queries_router)
api_router.include_router(connections_router)
api_router.include_router(projects_router)
