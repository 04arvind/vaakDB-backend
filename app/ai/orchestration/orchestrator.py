"""The single entry point used by api/routes.py.

    chat          -> LangGraph (orchestration/graph.py): route -> SQL | RAG | chat
    query         -> QA workflow (always text-to-SQL over PostgreSQL)
    ask           -> RAG workflow (always ChromaDB knowledge base)
    ingest_text   -> extraction workflow -> registered PostgreSQL write target
    ingest_file   -> Gemini file reader -> extraction workflow
    add_document  -> knowledge base (ChromaDB) for later RAG

Every call is scoped to user_id. Conversations are persisted, and long-term
memory is updated in the background so it never slows down or fails a reply.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.ai.config import settings
from app.ai.db.vector_store import list_documents
from app.ai.memory.conversation import add_exchange, ensure_session, get_history
from app.ai.memory.long_term import update_long_term_memory
from app.ai.orchestration import aggregator
from app.ai.orchestration.graph import ChatState, build_chat_graph
from app.ai.orchestration.router import choose_provider
from app.ai.schemas.llm_io import Message
from app.ai.schemas.responses import (
    ChatResponse,
    DocumentOut,
    IngestResponse,
    QueryResponse,
)
from app.ai.workflows.index_workflow import delete_document, index_document
from app.ai.workflows.ingest_workflow import run_ingest
from app.ai.workflows.multimodal_workflow import run_file_ingest
from app.ai.workflows.qa_workflow import run_qa
from app.ai.workflows.rag_workflow import run_rag

log = logging.getLogger(__name__)

MEMORY_MIN_CHARS = 20       # don't run memory extraction on "hi" / "thanks"
MEMORY_WINDOW = 6           # recent messages the memory agent looks at
SHUTDOWN_GRACE_S = 10.0
NO_DOCS_ANSWER = "I could not find anything relevant in your documents."


class Orchestrator:
    def __init__(self) -> None:
        self._background: set[asyncio.Task] = set()
        self._graph: Any = None

    @property
    def graph(self) -> Any:
        if self._graph is None:
            self._graph = build_chat_graph(on_saved=self._remember_state)
        return self._graph

    # ------------------------------------------------------------------ chat
    async def chat(
        self,
        *,
        message: str,
        session_id: str | None,
        user_id: str,
        provider: str | None = None,
    ) -> ChatResponse:
        state: ChatState = await self.graph.ainvoke(
            {"message": message, "user_id": user_id, "session_id": session_id,
             "provider": provider}
        )
        return aggregator.chat_response(
            state["session_id"], state["answer"], route=state["answered_by"],
            provider=state.get("provider_used"), model=state.get("model_used"),
            usage=state["usage"], sql=state.get("sql"), sources=state.get("sources") or [],
        )

    # ----------------------------------------------------------------- query
    async def query(
        self,
        *,
        question: str,
        session_id: str | None,
        user_id: str,
        include_rows: bool = False,
    ) -> QueryResponse:
        sid = await ensure_session(user_id, session_id)
        history = await get_history(sid, user_id, limit=settings.HISTORY_LIMIT)
        qa = await run_qa(question, history=history)
        meta = aggregator.exchange_meta(
            route="data_query", provider=qa.provider, model=qa.model, usage=qa.usage,
            sql=qa.sql,
        )
        await add_exchange(sid, user_id, question, qa.answer, meta=meta)
        return aggregator.query_response(sid, qa, include_rows=include_rows)

    # ------------------------------------------------------------------- ask
    async def ask(
        self, *, question: str, session_id: str | None, user_id: str
    ) -> ChatResponse:
        """Answer strictly from the user's knowledge base (no SQL, no general chat)."""
        sid = await ensure_session(user_id, session_id)
        history = await get_history(sid, user_id, limit=settings.HISTORY_LIMIT)
        result = await run_rag(question, user_id=user_id, history=history)
        if result is None:
            answer, sources, provider, model, usage = NO_DOCS_ANSWER, [], None, None, None
        else:
            answer, sources = result.answer, [vars(s) for s in result.sources]
            provider, model, usage = result.provider, result.model, result.usage
        response = aggregator.chat_response(
            sid, answer, route="knowledge", provider=provider, model=model,
            usage=usage, sources=sources,
        )
        meta = aggregator.exchange_meta(route="knowledge", provider=provider, model=model,
                                        usage=response.usage)
        await add_exchange(sid, user_id, question, answer, meta=meta)
        return response

    # ------------------------------------------------------- knowledge base
    async def add_document(
        self,
        *,
        user_id: str,
        content: bytes | None = None,
        text: str | None = None,
        mime_type: str = "text/plain",
        filename: str = "document",
        title: str | None = None,
    ) -> DocumentOut:
        result = await index_document(user_id=user_id, content=content, text=text,
                                      mime_type=mime_type, filename=filename, title=title)
        return DocumentOut(doc_id=result.doc_id, filename=filename, title=title or filename,
                           mime_type=mime_type, chunks=result.chunks)

    async def list_documents(self, *, user_id: str) -> list[DocumentOut]:
        docs = await list_documents("documents", owner=user_id)
        return [DocumentOut(**d) for d in docs]

    async def delete_document(self, *, user_id: str, doc_id: str) -> bool:
        return await delete_document(user_id=user_id, doc_id=doc_id)

    # ---------------------------------------------------------------- ingest
    async def ingest_text(
        self, *, text: str, target: str, user_id: str, dry_run: bool = False
    ) -> IngestResponse:
        result = await run_ingest(
            text, target, user_id=user_id, dry_run=dry_run,
            provider=choose_provider("extract", text_len=len(text)),
        )
        return aggregator.ingest_response(result)

    async def ingest_file(
        self,
        *,
        content: bytes,
        mime_type: str,
        filename: str,
        target: str,
        user_id: str,
        dry_run: bool = False,
    ) -> IngestResponse:
        result = await run_file_ingest(
            content, mime_type=mime_type, filename=filename, target=target,
            user_id=user_id, dry_run=dry_run,
        )
        return aggregator.ingest_response(result.ingest, source_text=result.source_text)

    # --------------------------------------------------------------- memory
    def _remember_state(self, state: ChatState) -> None:
        self._remember(state["user_id"], state["session_id"], state.get("history", []),
                       state["message"], state["answer"])

    def _remember(self, user_id: str, session_id: str, history: list[Message],
                  user_text: str, answer: str) -> None:
        if not settings.MEMORY_ENABLED or len(user_text) < MEMORY_MIN_CHARS:
            return
        window = [
            *history[-(MEMORY_WINDOW - 2):],
            Message(role="user", content=user_text),
            Message(role="assistant", content=answer),
        ]
        task = asyncio.create_task(update_long_term_memory(user_id, session_id, window))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def shutdown(self) -> None:
        """Let in-flight memory updates finish (bounded) before the process exits."""
        if not self._background:
            return
        _, pending = await asyncio.wait(set(self._background), timeout=SHUTDOWN_GRACE_S)
        for task in pending:
            task.cancel()


orchestrator = Orchestrator()
