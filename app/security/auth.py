"""JWT access tokens.

Tokens carry `sub` (user id), signed with JWT_SECRET (falls back to
AI_JWT_SECRET). Every request re-loads the user from meta_db, so a deactivated
account stops working immediately.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from jose import JWTError, jwt

from app.core.config import settings
from app.core.exceptions import AppError, ErrorCode
from app.security.secrets import generate_secret

log = logging.getLogger(__name__)

_ephemeral_secret: str | None = None


def _secret() -> str:
    global _ephemeral_secret
    if settings.jwt_secret and settings.jwt_secret.get_secret_value().strip():
        return settings.jwt_secret.get_secret_value().strip()
    if settings.environment == "production":
        raise AppError(ErrorCode.NOT_CONFIGURED, "Authentication is not configured.")
    if _ephemeral_secret is None:
        log.warning("JWT_SECRET is not set: using a random secret; tokens stop working on restart")
        _ephemeral_secret = generate_secret()
    return _ephemeral_secret


def create_access_token(user_id: int) -> tuple[str, int]:
    expires_in = settings.access_token_expire_minutes * 60
    now = datetime.now(UTC)
    claims = {"sub": str(user_id), "iat": int(now.timestamp()),
              "exp": int((now + timedelta(seconds=expires_in)).timestamp()), "typ": "access"}
    return jwt.encode(claims, _secret(), algorithm=settings.jwt_algorithm), expires_in


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        claims = jwt.decode(token, _secret(), algorithms=[settings.jwt_algorithm])
    except JWTError:
        raise AppError(ErrorCode.NOT_AUTHENTICATED, "Invalid or expired token.") from None
    if claims.get("typ") != "access" or not str(claims.get("sub", "")).isdigit():
        raise AppError(ErrorCode.NOT_AUTHENTICATED, "Invalid token.")
    return claims
