"""Server / database connection status. Contains no user data and never returns
URLs or passwords, so any signed-in user may read it."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.api.deps import CurrentCaller
from app.schemas.connection import ConnectionStatus
from app.services import connection_service

router = APIRouter(prefix="/connections", tags=["Connections"])


@router.get("/status", response_model=ConnectionStatus)
def connection_status(caller: CurrentCaller) -> dict[str, Any]:
    return connection_service.status()
