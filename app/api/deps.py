"""Shared FastAPI dependencies: database session, current user, caller (workspace)."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.exceptions import AppError, ErrorCode
from app.db.models import User
from app.db.session import get_db
from app.security.auth import decode_access_token
from app.security.authorization import Caller

DbSession = Annotated[Session, Depends(get_db)]
_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    db: DbSession,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> User:
    if creds is None or not creds.credentials:
        raise AppError(ErrorCode.NOT_AUTHENTICATED)
    claims = decode_access_token(creds.credentials)
    user = db.get(User, int(claims["sub"]))
    if user is None or not user.is_active:
        raise AppError(ErrorCode.NOT_AUTHENTICATED, "Invalid or expired token.")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def get_caller(user: CurrentUser) -> Caller:
    """Every request runs inside the signed-in user's own workspace."""
    return Caller(user_id=user.id)


CurrentCaller = Annotated[Caller, Depends(get_caller)]
