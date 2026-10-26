"""Request bodies for api/routes.py. Unknown fields are rejected."""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.ai.config import ProviderName

SessionId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
TargetName = Annotated[str, StringConstraints(pattern=r"^[a-z_][a-z0-9_]{0,62}$")]
NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ChatRequest(_Request):
    message: Annotated[NonBlank, Field(max_length=8_000)]
    session_id: SessionId | None = None
    provider: ProviderName | None = Field(
        default=None, description="Force a provider; normally the router decides."
    )


class QueryRequest(_Request):
    question: Annotated[NonBlank, Field(max_length=2_000)]
    session_id: SessionId | None = None
    include_rows: bool = Field(default=False, description="Return the raw result rows too.")


class AskRequest(_Request):
    question: Annotated[NonBlank, Field(max_length=2_000)]
    session_id: SessionId | None = None


class DocumentTextRequest(_Request):
    text: Annotated[NonBlank, Field(max_length=500_000)]
    title: Annotated[str, StringConstraints(max_length=255)] = "note"


class IngestRequest(_Request):
    text: Annotated[NonBlank, Field(max_length=50_000)]
    target: TargetName
    dry_run: bool = Field(default=False, description="Extract only; write nothing.")
