from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.db import app_db
from app.services.metadata_ops import quick_health

router = APIRouter()


@router.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "VaakDB Backend",
    }


@router.get("/health/ready", tags=["Health"])
def readiness() -> JSONResponse:
    """Both databases reachable."""
    checks = {"meta_db": app_db.ping(), "target_db": quick_health()}
    ok = all(checks.values())
    return JSONResponse({"status": "ready" if ok else "degraded", "checks": checks},
                        status_code=200 if ok else 503)
