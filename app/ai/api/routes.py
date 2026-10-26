"""HTTP layer for the AI module.

Rules for this file:
- Validate input, call the orchestrator, translate errors to HTTP codes.
- NO business logic and NO direct provider/DB calls live here (only the
  thin session/memory listing helpers, which are pure ownership-scoped reads).

Mount it in your FastAPI app:
    from app.ai.api.routes import router as ai_router
    from app.ai.lifecycle import lifespan
    app = FastAPI(lifespan=lifespan)
    app.include_router(ai_router)
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Annotated, Generator

from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, Query, UploadFile, status
from pydantic import ValidationError

from app.ai.agents import AgentError, CannotAnswer
from app.ai.config import settings
from app.ai.db.engine import get_read_engine, get_write_engine, ping
from app.ai.exceptions import (
    ConfigurationError,
    QueryFailedError,
    RateLimitExceeded,
    SessionNotFoundError,
    SQLGuardError,
    UnknownTargetError,
)
from app.ai.api.deps import LimitedUserId, UserId, request_context
from app.ai.memory import conversation, long_term
from app.ai.orchestration.orchestrator import orchestrator
from app.ai.providers.base import (
    ContentBlockedError,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderRateLimitError,
)
from app.ai.providers.registry import is_configured
from app.ai.schemas.requests import (
    AskRequest,
    ChatRequest,
    DocumentTextRequest,
    IngestRequest,
    QueryRequest,
    SessionId,
)
from app.ai.schemas.responses import (
    ChatResponse,
    DocumentOut,
    HealthResponse,
    HistoryResponse,
    IngestResponse,
    MemoryOut,
    QueryResponse,
    SessionSummary,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["ai"], dependencies=[Depends(request_context)])

# --------------------------------------------------------------------------- #
# Upload limits (multimodal ingest)
# --------------------------------------------------------------------------- #
MAX_UPLOAD_BYTES = settings.MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_UPLOAD_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "application/pdf",
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/mp4",
    "audio/webm",
}
# The knowledge base also takes plain-text formats (indexed without an LLM call).
DOCUMENT_UPLOAD_TYPES = ALLOWED_UPLOAD_TYPES | {
    "text/plain", "text/markdown", "text/csv", "application/json",
}

# Magic-byte signatures: the declared content type must match the file itself.
_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "application/pdf": (b"%PDF-",),
}


# Error translation
@contextmanager
def ai_errors() -> Generator[None, None, None]:
    """Map internal exceptions to clean HTTP errors without leaking internals."""
    try:
        yield
    except HTTPException:
        raise
    except CannotAnswer as exc:  # must come before AgentError (subclass)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except SQLGuardError as exc:
        log.warning("SQL blocked by guard: %s", exc)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "The generated query was blocked by the safety guard.",
        ) from exc
    except QueryFailedError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except SessionNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except UnknownTargetError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except RateLimitExceeded as exc:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(exc)) from exc
    except ProviderRateLimitError as exc:  # must come before ProviderError
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "AI providers are rate-limited. Try again shortly.",
        ) from exc
    except ConfigurationError as exc:
        log.error("AI module misconfigured: %s", exc)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "The AI service is not configured."
        ) from exc
    except ProviderNotConfiguredError as exc:
        log.error("No AI provider available: %s", exc)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "The AI service is not configured."
        ) from exc
    except ContentBlockedError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "The request was declined by the AI provider's safety filters.",
        ) from exc
    except (ProviderError, AgentError) as exc:
        log.error("AI upstream failure: %s", exc)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "The AI service failed to produce a result."
        ) from exc
    except ValidationError as exc:  # pydantic; subclass of ValueError, keep details internal
        log.warning("Validation failed inside AI pipeline: %s", exc)
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "The data could not be validated."
        ) from exc
    except ValueError as exc:  # our own user-facing input errors
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except Exception as exc:
        log.exception("Unhandled AI error")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal server error."
        ) from exc


async def _read_limited(file: UploadFile) -> bytes:
    """Read an upload in chunks and abort early if it exceeds the size limit."""
    data = bytearray()
    while chunk := await file.read(1024 * 1024):
        data.extend(chunk)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"File exceeds {settings.MAX_UPLOAD_MB} MB limit.",
            )
    return bytes(data)


def _matches_signature(content: bytes, mime_type: str) -> bool:
    if mime_type == "image/webp":
        return content[:4] == b"RIFF" and content[8:12] == b"WEBP"
    signatures = _SIGNATURES.get(mime_type)
    return True if signatures is None else content.startswith(signatures)


# Endpoints
@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness: shows which providers are configured (never returns the keys)."""
    providers = {name: is_configured(name) for name in ("groq", "gemini")}
    return HealthResponse(status="ok" if any(providers.values()) else "degraded",
                          providers=providers)


@router.get("/ready", response_model=HealthResponse)
async def ready() -> HealthResponse:
    """Readiness: also checks both database connections."""
    providers = {name: is_configured(name) for name in ("groq", "gemini")}
    database: dict[str, bool] = {}
    for label, getter in (("read", get_read_engine), ("write", get_write_engine)):
        try:
            database[label] = await ping(getter())
        except RuntimeError:  # URL not configured
            database[label] = False
    ok = any(providers.values()) and all(database.values())
    result = HealthResponse(status="ok" if ok else "degraded", providers=providers,
                            database=database)
    if not ok:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, result.model_dump())
    return result


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, user_id: LimitedUserId) -> ChatResponse:
    """Multimodal-orchestrated conversation (LangGraph): each message is routed to
    text-to-SQL over PostgreSQL, RAG over the user's documents, or plain chat."""
    with ai_errors():
        return await orchestrator.chat(
            message=body.message,
            session_id=body.session_id,
            user_id=user_id,
            provider=body.provider,  # optional manual override, else router decides
        )


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest, user_id: LimitedUserId) -> QueryResponse:
    """Question -> SQL -> guarded read-only execution -> natural-language answer."""
    with ai_errors():
        return await orchestrator.query(
            question=body.question,
            session_id=body.session_id,
            user_id=user_id,
            include_rows=body.include_rows,
        )


@router.post("/ask", response_model=ChatResponse)
async def ask(body: AskRequest, user_id: LimitedUserId) -> ChatResponse:
    """RAG only: answer from the user's uploaded documents and indexed records, with sources."""
    with ai_errors():
        return await orchestrator.ask(
            question=body.question, session_id=body.session_id, user_id=user_id
        )


# ---- knowledge base (ChromaDB) --------------------------------------------- #
@router.post("/documents", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    user_id: LimitedUserId,
    file: Annotated[UploadFile, File(description="PDF, image, audio or text file")],
    title: Annotated[str | None, Form(max_length=255)] = None,
) -> DocumentOut:
    """Add a file to the user's knowledge base. Non-text files are transcribed by Gemini."""
    mime_type = (file.content_type or "").split(";")[0].strip().lower()
    if mime_type not in DOCUMENT_UPLOAD_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type: {mime_type or 'unknown'}",
        )
    content = await _read_limited(file)
    if not content:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty.")
    if not _matches_signature(content, mime_type):
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "File content does not match its type."
        )
    with ai_errors():
        return await orchestrator.add_document(
            user_id=user_id, content=content, mime_type=mime_type,
            filename=(file.filename or "upload")[:255], title=title,
        )


@router.post("/documents/text", response_model=DocumentOut,
             status_code=status.HTTP_201_CREATED)
async def add_text_document(body: DocumentTextRequest, user_id: LimitedUserId) -> DocumentOut:
    """Add pasted text (notes, policies, FAQs) to the user's knowledge base."""
    with ai_errors():
        return await orchestrator.add_document(
            user_id=user_id, text=body.text, filename=body.title, title=body.title
        )


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents(user_id: UserId) -> list[DocumentOut]:
    with ai_errors():
        return await orchestrator.list_documents(user_id=user_id)


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    user_id: UserId, doc_id: Annotated[str, Path(pattern=r"^[a-f0-9]{32}$")]
) -> None:
    with ai_errors():
        deleted = await orchestrator.delete_document(user_id=user_id, doc_id=doc_id)
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")


# ---- structured ingest (PostgreSQL write targets) -------------------------- #
@router.post("/ingest", response_model=IngestResponse)
async def ingest(body: IngestRequest, user_id: LimitedUserId) -> IngestResponse:
    """Raw text -> structured extraction -> validated insert.

    Set dry_run=true to preview the extracted data without writing anything.
    """
    with ai_errors():
        return await orchestrator.ingest_text(
            text=body.text,
            target=body.target,
            user_id=user_id,
            dry_run=body.dry_run,
        )


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    user_id: LimitedUserId,
    file: Annotated[UploadFile, File(description="Image, PDF or audio file")],
    target: Annotated[
        str, Form(pattern=r"^[a-z_][a-z0-9_]{0,62}$", description="Target, e.g. 'expenses'")
    ],
    dry_run: Annotated[bool, Form()] = False,
) -> IngestResponse:
    """Multimodal ingest: the file is read by Gemini, then stored the same way as
    text ingest."""
    mime_type = (file.content_type or "").split(";")[0].strip().lower()
    if mime_type not in ALLOWED_UPLOAD_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type: {mime_type or 'unknown'}",
        )

    content = await _read_limited(file)
    if not content:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty.")
    if not _matches_signature(content, mime_type):
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "File content does not match its type."
        )

    with ai_errors():
        return await orchestrator.ingest_file(
            content=content,
            mime_type=mime_type,
            filename=(file.filename or "upload")[:255],
            target=target,
            user_id=user_id,
            dry_run=dry_run,
        )


# ---- conversations --------------------------------------------------------- #
@router.get("/sessions", response_model=list[SessionSummary])
async def list_sessions(
    user_id: UserId, limit: Annotated[int, Query(ge=1, le=200)] = 50
) -> list[SessionSummary]:
    with ai_errors():
        rows = await conversation.list_sessions(user_id, limit=limit)
    return [SessionSummary(**r) for r in rows]


@router.get("/sessions/{session_id}", response_model=HistoryResponse)
async def session_history(
    user_id: UserId,
    session_id: Annotated[SessionId, Path()],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> HistoryResponse:
    with ai_errors():
        rows = await conversation.get_history_records(session_id, user_id, limit=limit)
    if not rows:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found.")
    return HistoryResponse(session_id=session_id, messages=rows)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(user_id: UserId, session_id: Annotated[SessionId, Path()]) -> None:
    with ai_errors():
        deleted = await conversation.delete_session(session_id, user_id)
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found.")


# ---- long-term memory ------------------------------------------------------ #
@router.get("/memories", response_model=list[MemoryOut])
async def list_memories(
    user_id: UserId, limit: Annotated[int, Query(ge=1, le=500)] = 100
) -> list[MemoryOut]:
    with ai_errors():
        rows = await long_term.list_memories(user_id, limit=limit)
    return [MemoryOut(**r) for r in rows]


@router.delete("/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(user_id: UserId, memory_id: Annotated[int, Path(ge=1)]) -> None:
    with ai_errors():
        deleted = await long_term.delete_memory(user_id, memory_id)
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Memory not found.")


@router.delete("/memories", status_code=status.HTTP_204_NO_CONTENT)
async def delete_all_memories(user_id: UserId) -> None:
    with ai_errors():
        await long_term.delete_all_memories(user_id)
