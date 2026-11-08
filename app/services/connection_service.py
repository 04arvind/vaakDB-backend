"""Health / configuration status of the two databases (no URLs or passwords)."""
from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.db import app_db, target_db


def status() -> dict[str, Any]:
    ro = target_db.ping("ro")
    rw = target_db.ping("rw")
    warnings: list[str] = []
    if not settings.target_db_ro_url or not settings.target_db_rw_url:
        warnings.append("TARGET_DB_RO_URL / TARGET_DB_RW_URL not set: falling back to "
                        "TARGET_DATABASE_URL for both (development only).")
    if ro.get("superuser") or rw.get("superuser"):
        warnings.append("A target connection uses a superuser role. Use vaak_ro / vaak_rw "
                        "(scripts/create_ro_role.sql, scripts/create_rw_role.sql).")
    if ro.get("ok") and rw.get("ok") and ro.get("role") == rw.get("role"):
        warnings.append("Reads and writes use the same database role.")
    return {
        "meta_db": app_db.ping(),
        "target_read": ro,
        "target_write": rw,
        "default_schema": settings.default_schema,
        "ai_routes_enabled": settings.enable_ai_routes,
        "warnings": warnings,
    }
