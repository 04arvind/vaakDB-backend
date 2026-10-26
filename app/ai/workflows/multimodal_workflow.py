"""Image / PDF / audio -> text (Gemini) -> the normal text ingest workflow.

Two steps instead of one multimodal extraction call, so validation, retries,
dry-run and the write path are exactly the same as for text ingest, and the
transcribed text can be shown back to the user.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.ai.agents.file_reader_agent import FileReaderAgent
from app.ai.db.writer import get_target
from app.ai.workflows.ingest_workflow import IngestResult, run_ingest

MAX_SOURCE_PREVIEW = 5000


@dataclass
class FileIngestResult:
    ingest: IngestResult
    source_text: str


async def run_file_ingest(
    content: bytes,
    *,
    mime_type: str,
    filename: str,
    target: str,
    user_id: str,
    dry_run: bool = False,
    extract_provider: str | None = None,
) -> FileIngestResult:
    get_target(target)  # fail fast on an unknown target, before the expensive call

    reader = FileReaderAgent()  # multimodal: Gemini (fallback skips non-multimodal providers)
    text = await reader.read(content, mime_type, filename)

    result = await run_ingest(
        text, target, user_id=user_id, dry_run=dry_run, provider=extract_provider
    )
    result.usage = result.usage + reader.usage
    return FileIngestResult(ingest=result, source_text=text[:MAX_SOURCE_PREVIEW])
