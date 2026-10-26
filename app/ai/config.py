"""Settings for the AI module, read from environment variables / .env.

List settings accept either JSON (["a","b"]) or a comma-separated string (a,b).
Secrets are SecretStr so they never appear in logs or reprs.

Minimal .env for local development:
    GROQ_API_KEY=...
    GEMINI_API_KEY=...
    DATABASE_READ_URL=postgresql://ai_reader:pw@localhost:5432/vaakdb
    DATABASE_WRITE_URL=postgresql://ai_writer:pw@localhost:5432/vaakdb
    AI_JWT_SECRET=...            # or AI_ALLOW_ANONYMOUS=true for local testing only
"""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ProviderName = Literal["groq", "gemini"]


def _split(value: Any) -> Any:
    """Accept "a, b" as well as a JSON list for list-like settings."""
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("["):
            import json

            return json.loads(value)
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


class AISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- environment ------------------------------------------------------- #
    AI_ENV: Literal["development", "test", "production"] = "development"
    AI_LOG_LEVEL: str = "INFO"
    AI_LOG_JSON: bool = False

    # ---- providers --------------------------------------------------------- #
    GROQ_API_KEY: SecretStr | None = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

    GEMINI_API_KEY: SecretStr | None = None
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_EMBED_MODEL: str = "gemini-embedding-001"
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta"
    # 0 disables "thinking" on Flash models so it cannot eat the output budget.
    # Set to empty / -1 for models that do not accept a thinking budget.
    GEMINI_THINKING_BUDGET: int | None = 0

    LLM_TIMEOUT_S: float = Field(default=60.0, gt=0)
    LLM_MAX_RETRIES: int = Field(default=2, ge=0, le=6)
    # Order tried when the preferred provider fails or is not configured.
    PROVIDER_FALLBACK_ORDER: Annotated[list[ProviderName], NoDecode] = ["groq", "gemini"]
    # Inputs longer than this (characters) are routed to Gemini (long context).
    LONG_CONTEXT_CHARS: int = Field(default=24_000, gt=0)

    # ---- database ---------------------------------------------------------- #
    DATABASE_READ_URL: str | None = Field(default=None, repr=False)
    DATABASE_WRITE_URL: str | None = Field(default=None, repr=False)
    DB_SCHEMA: str = "public"
    DB_POOL_SIZE: int = Field(default=5, ge=1)
    DB_MAX_OVERFLOW: int = Field(default=5, ge=0)
    DB_POOL_RECYCLE_S: int = Field(default=1800, ge=60)

    SQL_TIMEOUT_MS: int = Field(default=5000, ge=100, le=120_000)
    WRITE_TIMEOUT_MS: int = Field(default=15_000, ge=100, le=300_000)
    SQL_MAX_ROWS: int = Field(default=100, ge=1, le=10_000)
    SQL_MAX_REPAIRS: int = Field(default=2, ge=0, le=5)

    SCHEMA_CACHE_TTL: int = Field(default=300, ge=0)
    # Schema text above this many tokens is narrowed to tables relevant to the question.
    SCHEMA_MAX_TOKENS: int = Field(default=3000, ge=200)
    AI_HIDDEN_TABLES: Annotated[set[str], NoDecode] = set()
    AI_HIDDEN_PREFIXES: Annotated[tuple[str, ...], NoDecode] = ("ai_",)
    # When non-empty, ONLY these tables are visible to the SQL agent.
    AI_READABLE_TABLES: Annotated[set[str], NoDecode] = set()

    EMBEDDING_DIM: int = Field(default=768, ge=8, le=4096)

    # ---- vector store (ChromaDB) / RAG ------------------------------------- #
    # "persistent": embedded, files under CHROMA_PATH (single server).
    # "http": a Chroma server shared by several app instances (production).
    CHROMA_MODE: Literal["persistent", "http", "memory"] = "persistent"
    CHROMA_PATH: str = "./.chroma"
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_SSL: bool = False
    CHROMA_AUTH_TOKEN: SecretStr | None = None
    CHROMA_COLLECTION_PREFIX: str = "vaakdb_"

    RAG_TOP_K: int = Field(default=6, ge=1, le=50)
    RAG_MIN_SCORE: float = Field(default=0.35, ge=0.0, le=1.0)  # cosine similarity
    RAG_MAX_CONTEXT_TOKENS: int = Field(default=3000, ge=200)
    CHUNK_SIZE: int = Field(default=1200, ge=200, le=8000)  # characters
    CHUNK_OVERLAP: int = Field(default=150, ge=0, le=2000)
    KB_MAX_CHUNKS_PER_DOC: int = Field(default=2000, ge=1)
    # Tables indexed into Chroma by workflows/index_workflow.py (optional).
    RAG_INDEX_TABLES: Annotated[set[str], NoDecode] = set()

    # ---- memory / context -------------------------------------------------- #
    AI_AUTO_CREATE_TABLES: bool = False
    MEMORY_ENABLED: bool = True
    MEMORY_MAX_PER_USER: int = Field(default=200, ge=1)
    HISTORY_LIMIT: int = Field(default=20, ge=0, le=200)
    CONTEXT_MAX_TOKENS: int = Field(default=6000, ge=1000)

    # ---- API --------------------------------------------------------------- #
    AI_JWT_SECRET: SecretStr | None = None
    AI_JWT_ALGORITHM: str = "HS256"
    AI_JWT_AUDIENCE: str | None = None
    AI_JWT_USER_CLAIM: str = "sub"
    # Lets requests without a token through as "anonymous". Never in production.
    AI_ALLOW_ANONYMOUS: bool = False

    RATE_LIMIT_PER_MINUTE: int = Field(default=30, ge=0)  # 0 disables
    RATE_LIMIT_BURST: int = Field(default=10, ge=1)
    MAX_UPLOAD_MB: int = Field(default=14, ge=1, le=19)  # Gemini inline limit is ~20 MB

    @field_validator(
        "PROVIDER_FALLBACK_ORDER", "AI_HIDDEN_TABLES", "AI_HIDDEN_PREFIXES",
        "AI_READABLE_TABLES", "RAG_INDEX_TABLES", mode="before",
    )
    @classmethod
    def _parse_lists(cls, value: Any) -> Any:
        return _split(value)

    @field_validator("AI_HIDDEN_TABLES", "AI_READABLE_TABLES", "RAG_INDEX_TABLES", mode="after")
    @classmethod
    def _lower_set(cls, value: set[str]) -> set[str]:
        return {v.lower() for v in value}

    @field_validator("GEMINI_THINKING_BUDGET", mode="before")
    @classmethod
    def _thinking(cls, value: Any) -> Any:
        if value in ("", None) or (str(value).strip() == "-1"):
            return None
        return value

    @model_validator(mode="after")
    def _production_checks(self) -> "AISettings":
        if self.AI_ENV == "production" and self.AI_ALLOW_ANONYMOUS:
            raise ValueError("AI_ALLOW_ANONYMOUS must be false when AI_ENV=production.")
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        return self

    # ---- helpers ----------------------------------------------------------- #
    def api_key(self, provider: str) -> str | None:
        secret = {"groq": self.GROQ_API_KEY, "gemini": self.GEMINI_API_KEY}.get(provider)
        value = secret.get_secret_value().strip() if secret else ""
        return value or None


@lru_cache
def get_settings() -> AISettings:
    return AISettings()


settings = get_settings()
