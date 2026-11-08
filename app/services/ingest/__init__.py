"""File ingestion: upload storage + table extraction (PDF, CSV, XLSX)."""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import AppError, ErrorCode
from app.services.ingest.models import ExtractedTable, Extraction
from app.services.ingest.storage import CSV, PDF, XLSX

__all__ = ["ExtractedTable", "Extraction", "extract_file", "get_cached_extraction"]

log = logging.getLogger(__name__)

_CACHE_SIZE = 16
_cache: OrderedDict[str, Extraction] = OrderedDict()
_lock = threading.Lock()


def extract_file(path: Path, mime: str) -> Extraction:
    """Dispatch on the verified MIME type. Unexpected parser failures become FILE_UNREADABLE."""
    from app.services.ingest.pdf_extractor import extract_pdf
    from app.services.ingest.tabular import extract_csv, extract_xlsx

    try:
        if mime == PDF:
            return extract_pdf(path, max_pages=settings.max_pdf_pages)
        if mime == CSV:
            return extract_csv(path)
        if mime == XLSX:
            return extract_xlsx(path)
    except AppError:
        raise
    except Exception as exc:
        log.warning("Extraction failed for %s: %s", mime, type(exc).__name__)
        raise AppError(ErrorCode.FILE_UNREADABLE, "The file could not be read.") from None
    raise AppError(ErrorCode.UNSUPPORTED_FILE_TYPE)


def get_cached_extraction(sha256: str, path: Path, mime: str) -> Extraction:
    """Extraction is deterministic, so it is cached by content hash."""
    with _lock:
        if sha256 in _cache:
            _cache.move_to_end(sha256)
            return _cache[sha256]
    result = extract_file(path, mime)
    with _lock:
        _cache[sha256] = result
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return result
