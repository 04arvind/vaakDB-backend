"""Provider-neutral types for talking to LLMs.

Agents build `Message` lists; every provider turns them into its own wire format
and returns an `LLMResponse`, so agents never depend on a vendor SDK.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant"]


class Attachment(BaseModel):
    """Binary input (image, PDF, audio) for multimodal providers."""

    model_config = ConfigDict(frozen=True)

    mime_type: str
    data: bytes = Field(repr=False)
    filename: str | None = None


class Message(BaseModel):
    role: Role
    content: str
    attachments: list[Attachment] = Field(default_factory=list)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class LLMResponse(BaseModel):
    text: str
    provider: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str | None = None
    latency_ms: int = 0

    @property
    def truncated(self) -> bool:
        """True when the model stopped because it hit max_tokens."""
        return (self.finish_reason or "").lower() in {"length", "max_tokens"}
