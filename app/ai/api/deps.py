"""FastAPI dependencies for the AI routes: auth, rate limiting, request ids.

Authentication: Bearer JWT verified with AI_JWT_SECRET (python-jose). The user
id comes from the AI_JWT_USER_CLAIM claim ("sub" by default). When the main app
has its own auth, plug it in without touching this module:

    app.dependency_overrides[current_user_id] = my_auth_dependency  # returns user id str
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.ai.config import settings
from app.ai.exceptions import RateLimitExceeded
from app.ai.utils.logger import request_id_var
from app.ai.utils.rate_limiter import RateLimiter

log = logging.getLogger(__name__)

ANONYMOUS_USER = "anonymous"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_bearer = HTTPBearer(auto_error=False)

limiter = RateLimiter(settings.RATE_LIMIT_PER_MINUTE, settings.RATE_LIMIT_BURST)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"}
    )


async def current_user_id(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> str:
    if creds is None:
        if settings.AI_ALLOW_ANONYMOUS:
            return ANONYMOUS_USER
        raise _unauthorized("Not authenticated.")

    secret = settings.AI_JWT_SECRET.get_secret_value() if settings.AI_JWT_SECRET else ""
    if not secret:
        log.error("AI_JWT_SECRET is not set; cannot verify tokens")
        raise _unauthorized("Authentication is not configured.")

    from jose import JWTError, jwt  # imported lazily: only needed when tokens are used

    try:
        claims = jwt.decode(
            creds.credentials,
            secret,
            algorithms=[settings.AI_JWT_ALGORITHM],
            audience=settings.AI_JWT_AUDIENCE,
            options={"verify_aud": bool(settings.AI_JWT_AUDIENCE)},
        )
    except JWTError as exc:
        raise _unauthorized("Invalid or expired token.") from exc

    user = claims.get(settings.AI_JWT_USER_CLAIM)
    if user is None or str(user).strip() == "":
        raise _unauthorized("Token has no user.")
    return str(user)[:128]


UserId = Annotated[str, Depends(current_user_id)]


async def rate_limit(user_id: UserId, request: Request) -> str:
    """Per-user limit for endpoints that call an LLM. Anonymous users are keyed by IP."""
    key = user_id
    if user_id == ANONYMOUS_USER:
        key = f"ip:{request.client.host if request.client else 'unknown'}"
    try:
        limiter.check(key)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Too many requests. Please slow down.",
            headers={"Retry-After": str(max(1, int(exc.retry_after + 0.999)))},
        ) from exc
    return user_id


LimitedUserId = Annotated[str, Depends(rate_limit)]


async def request_context(request: Request, response: Response) -> str:
    """Accept (or create) an X-Request-ID, expose it to logs and echo it back."""
    incoming = request.headers.get("x-request-id", "")
    rid = incoming if _REQUEST_ID.match(incoming) else uuid.uuid4().hex
    request_id_var.set(rid)
    response.headers["X-Request-ID"] = rid
    return rid
