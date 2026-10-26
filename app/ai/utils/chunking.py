"""Text chunking for RAG, using LangChain's RecursiveCharacterTextSplitter.

Splits on paragraphs, then lines, then sentences, then words, so chunks keep
semantic units together. Sizes are in characters (CHUNK_SIZE / CHUNK_OVERLAP).
"""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def split_text(text: str, *, chunk_size: int | None = None,
               chunk_overlap: int | None = None) -> list[str]:
    from app.ai.config import settings

    size = chunk_size or settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=min(overlap, size - 1),
        separators=_SEPARATORS,
        strip_whitespace=True,
    )
    return [c for c in splitter.split_text(text or "") if c.strip()]
