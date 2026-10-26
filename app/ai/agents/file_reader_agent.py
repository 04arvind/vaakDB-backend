"""Image / PDF / audio -> plain text, using a multimodal provider (Gemini)."""
from __future__ import annotations

from app.ai.agents.base import AgentError, BaseAgent
from app.ai.prompts.loader import load_prompt
from app.ai.schemas.llm_io import Attachment

FILE_READER_SYSTEM = load_prompt("file_reader")


class FileReaderAgent(BaseAgent):
    default_provider = "gemini"
    temperature = 0.0
    max_tokens = 8000

    async def read(self, content: bytes, mime_type: str, filename: str | None = None) -> str:
        if not content:
            raise AgentError("The file is empty.")
        attachment = Attachment(mime_type=mime_type, data=content, filename=filename)
        resp = await self._ask(
            FILE_READER_SYSTEM,
            "Transcribe the attached file.",
            attachments=[attachment],
        )
        text = resp.text.strip()
        if not text:
            raise AgentError("No readable content found in the file.")
        return text
