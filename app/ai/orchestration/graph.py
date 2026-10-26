"""LangGraph state machine behind /ai/chat.

    START ─► load_context ─► route ─┬─ data_query ─► sql ──────answered──────────┐
                                    │                 └─cannot answer─┐            │
                                    ├─ knowledge ───────────────► retrieve        │
                                    ├─ general (not small talk) ─►    │            │
                                    │                       hits ─────┴─► rag ────►│
                                    │                       no hits ─► general ───►│
                                    └─ greeting ─────────────────────► general ───►└─► persist ─► END

- sql      : text-to-SQL over PostgreSQL (guarded, read-only) via workflows/qa_workflow
- retrieve : ChromaDB search over the user's uploaded documents + indexed DB rows
- rag      : answer from retrieved passages with [n] citations
- general  : plain chat with long-term memories in the prompt
- persist  : save the exchange, update long-term memory in the background

Every node is scoped by state["user_id"]; nothing in the graph trusts ids from
the model. Node failures in optional steps (schema, retrieval, memory) degrade
to the next best path instead of failing the request.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.ai.agents.chat_agent import CHAT_SYSTEM, ChatAgent
from app.ai.agents.sql_agent import CannotAnswer
from app.ai.config import settings
from app.ai.db.schema_inspector import get_table_names
from app.ai.db.vector_store import SearchHit
from app.ai.memory.context_builder import build_context, format_bullets
from app.ai.memory.conversation import add_exchange, ensure_session, get_history
from app.ai.memory.long_term import recall
from app.ai.orchestration import aggregator
from app.ai.orchestration.router import GREETING_REASON, choose_provider, route_message
from app.ai.schemas.llm_io import Message, Usage
from app.ai.workflows.qa_workflow import run_qa
from app.ai.workflows.rag_workflow import retrieve, run_rag

log = logging.getLogger(__name__)

# A "general" message only switches to RAG when retrieval is clearly relevant.
GENERAL_RAG_BONUS = 0.15


class ChatState(TypedDict, total=False):
    # input
    message: str
    user_id: str
    session_id: str | None
    provider: str | None
    # context
    history: list[Message]
    tables: list[str]
    # routing
    route: Literal["general", "data_query", "knowledge"]
    route_reason: str
    hits: list[SearchHit]
    # output
    answer: str
    answered_by: Literal["general", "data_query", "knowledge"]
    sql: str | None
    sources: list[dict[str, Any]]
    provider_used: str | None
    model_used: str | None
    usage: Usage


# --------------------------------------------------------------------------- #
# nodes
# --------------------------------------------------------------------------- #
async def load_context(state: ChatState) -> ChatState:
    sid = await ensure_session(state["user_id"], state.get("session_id"))
    history = await get_history(sid, state["user_id"], limit=settings.HISTORY_LIMIT)
    try:
        tables = await get_table_names()
    except Exception as exc:  # DB down / not configured: documents and chat still work
        log.warning("Schema unavailable (%s)", exc)
        tables = []
    return {"session_id": sid, "history": history, "tables": tables, "usage": Usage()}


async def route(state: ChatState) -> ChatState:
    decision = await route_message(state["message"], state["tables"],
                                   provider=state.get("provider"))
    log.info("Routed to %s (%.2f, %s)", decision.route, decision.confidence, decision.reason)
    return {"route": decision.route, "route_reason": decision.reason}


async def sql_node(state: ChatState) -> ChatState:
    try:
        qa = await run_qa(state["message"], history=state["history"],
                          provider=state.get("provider"))
    except CannotAnswer as exc:
        log.info("SQL path cannot answer (%s); trying documents", exc)
        return {"answer": ""}
    return {
        "answer": qa.answer, "answered_by": "data_query", "sql": qa.sql,
        "provider_used": qa.provider, "model_used": qa.model,
        "usage": state["usage"] + qa.usage,
    }


async def retrieve_node(state: ChatState) -> ChatState:
    try:
        hits = await retrieve(state["message"], user_id=state["user_id"],
                              history=state["history"])
    except Exception as exc:  # vector store down: fall back to plain chat
        log.warning("Retrieval failed (%s)", exc)
        hits = []
    if state.get("route") == "general":
        floor = settings.RAG_MIN_SCORE + GENERAL_RAG_BONUS
        hits = [h for h in hits if h.score >= floor]
    return {"hits": hits}


async def rag_node(state: ChatState) -> ChatState:
    result = await run_rag(state["message"], user_id=state["user_id"],
                           history=state["history"], hits=state["hits"],
                           provider=state.get("provider"))
    if result is None:
        return {"answer": ""}
    return {
        "answer": result.answer, "answered_by": "knowledge",
        "sources": [vars(s) for s in result.sources],
        "provider_used": result.provider, "model_used": result.model,
        "usage": state["usage"] + result.usage,
    }


async def general_node(state: ChatState) -> ChatState:
    memories: list[str] = []
    if settings.MEMORY_ENABLED:
        try:
            memories = await recall(state["user_id"], state["message"], limit=5)
        except Exception as exc:
            log.warning("Memory recall failed: %s", exc)
    ctx = build_context(
        system_prompt=CHAT_SYSTEM,
        user_message=state["message"],
        history=state["history"],
        sections=[("What you know about this user", format_bullets(memories))],
        max_tokens=settings.CONTEXT_MAX_TOKENS,
    )
    agent = ChatAgent(provider=choose_provider("chat", text_len=len(state["message"]),
                                               override=state.get("provider")))
    resp = await agent.reply(ctx.user, history=ctx.history, system=ctx.system)
    answer = resp.text.strip() or "Sorry, I could not produce an answer. Please try again."
    return {
        "answer": answer, "answered_by": "general",
        "provider_used": resp.provider, "model_used": resp.model,
        "usage": state["usage"] + agent.usage,
    }


def make_persist_node(on_saved: Callable[[ChatState], None] | None) -> Callable:
    async def persist(state: ChatState) -> ChatState:
        meta = aggregator.exchange_meta(
            route=state["answered_by"], provider=state.get("provider_used"),
            model=state.get("model_used"), usage=state["usage"], sql=state.get("sql"),
        )
        if state.get("sources"):
            meta["sources"] = [s["source"] for s in state["sources"]]
        await add_exchange(state["session_id"], state["user_id"], state["message"],
                           state["answer"], meta=meta)
        if on_saved is not None:
            on_saved(state)
        return {}

    return persist


# --------------------------------------------------------------------------- #
# edges
# --------------------------------------------------------------------------- #
def after_route(state: ChatState) -> str:
    if state["route"] == "data_query":
        return "sql"
    if state["route"] == "general" and state.get("route_reason") == GREETING_REASON:
        return "general"
    return "retrieve"


def after_sql(state: ChatState) -> str:
    return "persist" if state.get("answer") else "retrieve"


def after_retrieve(state: ChatState) -> str:
    return "rag" if state.get("hits") else "general"


def after_rag(state: ChatState) -> str:
    return "persist" if state.get("answer") else "general"


def build_chat_graph(on_saved: Callable[[ChatState], None] | None = None):
    """Compile the chat graph. `on_saved` runs after persistence (memory updates)."""
    graph = StateGraph(ChatState)
    graph.add_node("load_context", load_context)
    graph.add_node("route", route)
    graph.add_node("sql", sql_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rag", rag_node)
    graph.add_node("general", general_node)
    graph.add_node("persist", make_persist_node(on_saved))

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "route")
    graph.add_conditional_edges("route", after_route,
                                {"sql": "sql", "retrieve": "retrieve", "general": "general"})
    graph.add_conditional_edges("sql", after_sql, {"persist": "persist", "retrieve": "retrieve"})
    graph.add_conditional_edges("retrieve", after_retrieve, {"rag": "rag", "general": "general"})
    graph.add_conditional_edges("rag", after_rag, {"persist": "persist", "general": "general"})
    graph.add_edge("general", "persist")
    graph.add_edge("persist", END)
    return graph.compile()
