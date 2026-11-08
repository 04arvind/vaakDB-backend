"""Accounts. Every account is an independent user with its own workspace; there
are no roles and no endpoint that lists or changes other users."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Form, status
from sqlalchemy import func, select

from app.api.deps import CurrentUser, DbSession
from app.core.config import settings
from app.core.exceptions import AppError, ErrorCode
from app.db.models import User
from app.schemas.user import LoginRequest, RegisterRequest, TokenResponse, UserOut
from app.security.auth import create_access_token
from app.security.secrets import hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["Auth"])

# Verified against when the email is unknown, so timing does not reveal accounts.
_DUMMY_HASH = hash_password("timing-equalizer")


def _login(db: DbSession, email: str, password: str) -> TokenResponse:
    email = email.strip().lower()
    user = db.execute(select(User).where(func.lower(User.email) == email)).scalar_one_or_none()
    ok = verify_password(password, user.password_hash if user else _DUMMY_HASH)
    if user is None or not ok or not user.is_active:
        raise AppError(ErrorCode.INVALID_CREDENTIALS)
    token, expires_in = create_access_token(user.id)
    return TokenResponse(access_token=token, expires_in=expires_in)


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: DbSession) -> User:
    """Creates an account with its own, empty workspace."""
    if not settings.allow_registration:
        raise AppError(ErrorCode.REGISTRATION_DISABLED, "Registration is disabled.")
    exists = db.execute(select(User.id).where(func.lower(User.email) == body.email.lower())).first()
    if exists:
        raise AppError(ErrorCode.EMAIL_TAKEN, "An account with this email already exists.")
    user = User(email=body.email.lower(), password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: DbSession) -> TokenResponse:
    return _login(db, body.email, body.password)


@router.post("/token", response_model=TokenResponse, include_in_schema=True)
def token(
    db: DbSession,
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
) -> TokenResponse:
    """OAuth2 password form (for Swagger's Authorize button). username = email."""
    return _login(db, username, password)


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> User:
    return user
