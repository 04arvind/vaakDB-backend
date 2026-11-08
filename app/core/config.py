"""Backend settings, read from environment variables / .env.

Two databases, never mixed:
- META_DATABASE_URL: app internals (users, uploads, operations, audit_logs, query_history)
- TARGET_DB_RO_URL / TARGET_DB_RW_URL: the user's data the chat operates on.
  RO connects as the read-only role (vaak_ro), RW as the restricted write role
  (vaak_rw). Both fall back to TARGET_DATABASE_URL for local development only.

List settings accept "a,b" or a JSON list.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app.db.urls import normalize_pg_url

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Schemas that can never be targeted, whatever ALLOWED_SCHEMAS says.
SYSTEM_SCHEMAS = frozenset({"pg_catalog", "information_schema", "pg_toast"})


def _split(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("["):
            return json.loads(value)
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    app_name: str = "VaakDB API"
    app_version: str = "0.1.0"
    environment: Literal["development", "test", "production"] = Field(
        default="development", validation_alias=AliasChoices("APP_ENV", "ENVIRONMENT")
    )
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        validation_alias=AliasChoices("CORS_ORIGINS", "CLIENT_URL"),
    )

    # ---- databases --------------------------------------------------------- #
    meta_database_url: str = Field(
        validation_alias=AliasChoices("META_DATABASE_URL", "DATABASE_URL"), repr=False
    )
    target_database_url: str | None = Field(
        default=None, validation_alias=AliasChoices("TARGET_DATABASE_URL"), repr=False
    )
    target_db_ro_url: str | None = Field(
        default=None, validation_alias=AliasChoices("TARGET_DB_RO_URL"), repr=False
    )
    target_db_rw_url: str | None = Field(
        default=None, validation_alias=AliasChoices("TARGET_DB_RW_URL"), repr=False
    )
    # Every user gets a private workspace: their schemas are stored as
    # u<user_id>_<name>. This is the schema used when a request names none.
    default_schema: str = Field(default="main", validation_alias=AliasChoices("DEFAULT_SCHEMA"))
    max_schemas_per_user: int = Field(default=25, ge=1, le=1000)
    read_timeout_ms: int = Field(default=5_000, ge=100, le=120_000)
    statement_timeout_ms: int = Field(default=30_000, ge=100, le=600_000)
    lock_timeout_ms: int = Field(default=5_000, ge=100, le=120_000)

    # ---- operations / ingest ----------------------------------------------- #
    max_insert_rows: int = Field(default=10_000, ge=1, le=1_000_000)
    insert_batch_size: int = Field(default=500, ge=1, le=10_000)
    max_upload_mb: int = Field(default=10, ge=1, le=100)
    upload_dir: str = "./var/uploads"
    operation_ttl_minutes: int = Field(default=15, ge=1, le=24 * 60)
    max_pdf_pages: int = Field(default=50, ge=1, le=1000)
    llm_sample_rows: int = Field(default=30, ge=1, le=200)
    preview_rows: int = Field(default=10, ge=1, le=100)
    page_size_max: int = Field(default=200, ge=1, le=5_000)
    csv_export_max_rows: int = Field(default=50_000, ge=1, le=5_000_000)
    describe_sample_rows: int = Field(default=5, ge=0, le=50)

    # ---- LLM --------------------------------------------------------------- #
    llm_provider: Literal["groq", "gemini"] = "groq"

    # ---- auth -------------------------------------------------------------- #
    # Same secret as app/ai (AI_JWT_SECRET) so one token works for both routers.
    jwt_secret: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("JWT_SECRET", "AI_JWT_SECRET")
    )
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(default=60, ge=1, le=60 * 24 * 30)
    allow_registration: bool = True
    # The original app/ai routes (/ai/*) query one shared schema and are not
    # isolated per user, so they are off unless explicitly enabled.
    enable_ai_routes: bool = False

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _lists(cls, value: Any) -> Any:
        return _split(value)

    @field_validator("default_schema", mode="after")
    @classmethod
    def _default_schema(cls, value: str) -> str:
        value = value.strip().lower()
        if not value or value in SYSTEM_SCHEMAS or value.startswith("pg_"):
            raise ValueError("DEFAULT_SCHEMA must be a normal schema name")
        return value

    @model_validator(mode="after")
    def _checks(self) -> Settings:
        if self.environment == "production":
            if not self.jwt_secret or len(self.jwt_secret.get_secret_value()) < 32:
                raise ValueError("JWT_SECRET (>= 32 chars) is required in production.")
            if not (self.target_db_ro_url and self.target_db_rw_url):
                raise ValueError("TARGET_DB_RO_URL and TARGET_DB_RW_URL are required in production.")
        return self

    # ---- derived ----------------------------------------------------------- #
    @property
    def database_url(self) -> str:
        """Meta database (kept under this name for alembic/env.py)."""
        return normalize_pg_url(self.meta_database_url)

    @property
    def target_ro_url(self) -> str | None:
        url = self.target_db_ro_url or self.target_database_url
        return normalize_pg_url(url) if url else None

    @property
    def target_rw_url(self) -> str | None:
        url = self.target_db_rw_url or self.target_database_url
        return normalize_pg_url(url) if url else None

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        path = Path(self.upload_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings = get_settings()
