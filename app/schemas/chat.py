from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=4000)
    upload_id: uuid.UUID | None = None
    conversation_id: uuid.UUID | None = None


class ConversationOut(BaseModel):
    conversation_id: str
    title: str | None
    created_at: str | None
