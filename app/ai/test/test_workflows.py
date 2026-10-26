"""Workflows and orchestration with fake providers and patched DB / vector store."""
from __future__ import annotations

import asyncio
from datetime import date

import pytest
from pydantic import BaseModel

from app.ai.agents.sql_agent import CannotAnswer
from app.ai.db import writer
from app.ai.db.reader import QueryResult
from app.ai.db.vector_store import SearchHit
from app.ai.exceptions import (
    QueryExecutionError,
    QueryFailedError,
    RateLimitExceeded,
    SQLGuardError,
    UnknownTargetError,
)
from app.ai.memory.context_builder import build_context
from app.ai.prompts.loader import load_prompt, render
from app.ai.schemas.llm_io import Message
from app.ai.utils.rate_limiter import RateLimiter
from app.ai.workflows import ingest_workflow, qa_workflow, rag_workflow


def run(coro):
    return asyncio.run(coro)


SCHEMA = "TABLE orders\n  - id: integer [PK]\n  - total: numeric"


@pytest.fixture
def qa_env(monkeypatch):
    """Patch schema + execution; returns the list of SQL strings executed."""
    executed: list[str] = []
    outcomes: list[object] = []

    async def fake_schema(only=None, *, question=None):
        return SCHEMA

    async def fake_run_select(sql, *, max_rows=None):
        executed.append(sql)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(qa_workflow, "get_schema_context", fake_schema)
    monkeypatch.setattr(qa_workflow, "run_select", fake_run_select)
    return executed, outcomes


# ---- QA (text-to-SQL) ------------------------------------------------------- #
def test_qa_happy_path(fake_providers, qa_env):
    groq, _ = fake_providers
    executed, outcomes = qa_env
    outcomes.append(QueryResult(sql="SELECT count(*) FROM orders LIMIT 101",
                                columns=["count"], rows=[{"count": 42}]))
    groq.replies = ['{"sql": "SELECT count(*) FROM orders;", "explanation": "counts"}',
                    "You have 42 orders."]

    result = run(qa_workflow.run_qa("how many orders?"))
    assert result.answer == "You have 42 orders."
    assert executed == ["SELECT count(*) FROM orders"]  # trailing ';' stripped
    assert result.rows == [{"count": 42}] and result.attempts == 1
    assert result.usage.total_tokens == 30
    system_prompt = groq.calls[0]["messages"][0].content
    assert SCHEMA in system_prompt and date.today().isoformat() in system_prompt


def test_qa_repairs_after_db_error(fake_providers, qa_env):
    groq, _ = fake_providers
    executed, outcomes = qa_env
    outcomes += [QueryExecutionError('column "totl" does not exist'),
                 QueryResult(sql="SELECT sum(total) FROM orders", columns=["sum"],
                             rows=[{"sum": 10}])]
    groq.replies = ['{"sql": "SELECT sum(totl) FROM orders"}',
                    '{"sql": "SELECT sum(total) FROM orders"}',
                    "Total is 10."]
    result = run(qa_workflow.run_qa("total revenue?"))
    assert result.attempts == 2 and len(executed) == 2
    assert 'column "totl" does not exist' in groq.calls[1]["messages"][-1].content


def test_qa_gives_up_after_max_repairs(fake_providers, qa_env):
    groq, _ = fake_providers
    _, outcomes = qa_env
    outcomes += [QueryExecutionError("boom")] * 2
    groq.replies = ['{"sql": "SELECT 1 FROM orders"}'] * 2
    with pytest.raises(QueryFailedError):
        run(qa_workflow.run_qa("q", max_repairs=1))


def test_qa_reraises_guard_block_after_repairs(fake_providers, qa_env):
    groq, _ = fake_providers
    _, outcomes = qa_env
    outcomes += [SQLGuardError("Table not allowed: users.")]
    groq.replies = ['{"sql": "SELECT * FROM users"}']
    with pytest.raises(SQLGuardError):
        run(qa_workflow.run_qa("q", max_repairs=0))


def test_qa_cannot_answer(fake_providers, qa_env):
    groq, _ = fake_providers
    groq.replies = ['{"sql": "", "explanation": "No weather table."}']
    with pytest.raises(CannotAnswer, match="weather"):
        run(qa_workflow.run_qa("weather tomorrow?"))


# ---- structured ingest ----------------------------------------------------- #
class Expense(BaseModel):
    vendor: str | None = None
    amount: float | None = None
    spent_on: date | None = None


@pytest.fixture
def expense_target():
    writer.register_target("test_expenses", table="expenses", model=Expense,
                           owner_column="user_id", hints="Amounts are in INR.")
    yield
    writer.unregister_target("test_expenses")


def test_ingest_dry_run_extracts_and_validates(fake_providers, expense_target):
    groq, _ = fake_providers
    groq.replies = ['{"items": [{"vendor": "Uber", "amount": 250, "spent_on": "2026-10-01"},'
                    ' {"vendor": null, "amount": null, "spent_on": null}]}']
    result = run(ingest_workflow.run_ingest("Uber 250 on 1 Oct", "test_expenses",
                                            user_id="u1", dry_run=True))
    assert result.records == [{"vendor": "Uber", "amount": 250.0, "spent_on": "2026-10-01"}]
    assert result.inserted_ids == []
    assert "Amounts are in INR." in groq.calls[0]["messages"][-1].content


def test_ingest_retries_invalid_json(fake_providers, expense_target, monkeypatch):
    groq, _ = fake_providers
    groq.replies = ['{"items": [{"amount": "lots"}]}',
                    '{"items": [{"vendor": "Ola", "amount": 99}]}']
    inserted = {}

    async def fake_insert(target, items, *, user_id):
        inserted.update(target=target, n=len(items), user_id=user_id)
        return [7]

    monkeypatch.setattr(ingest_workflow, "insert_records", fake_insert)
    result = run(ingest_workflow.run_ingest("Ola 99", "test_expenses", user_id="u1"))
    assert result.inserted_ids == [7]
    assert inserted == {"target": "test_expenses", "n": 1, "user_id": "u1"}
    assert "previous answer was invalid" in groq.calls[1]["messages"][-1].content


def test_ingest_unknown_target_fails_before_llm(fake_providers):
    groq, _ = fake_providers
    with pytest.raises(UnknownTargetError):
        run(ingest_workflow.run_ingest("x", "nope", user_id="u1"))
    assert groq.calls == []


def test_register_target_rejects_bad_identifiers():
    with pytest.raises(ValueError):
        writer.register_target("x", table="expenses; drop table users", model=Expense)
    with pytest.raises(ValueError):
        writer.register_target("x", table="expenses", model=Expense, columns=("nope",))


# ---- RAG ------------------------------------------------------------------- #
def _hit(content, score, **meta):
    return SearchHit(id="d:0", content=content, score=score, namespace="documents",
                     metadata={"doc_id": "d", "filename": "policy.pdf", **meta})


def test_rag_answers_with_sources(fake_providers):
    groq, _ = fake_providers
    groq.replies = ["Refunds take 5 days [1]."]
    hits = [_hit("Refunds are processed within 5 business days.", 0.82)]
    result = run(rag_workflow.run_rag("how long do refunds take?", user_id="u1", hits=hits))
    assert result.answer.startswith("Refunds take 5 days")
    assert result.sources[0].source == "policy.pdf" and result.sources[0].index == 1
    system_prompt = groq.calls[0]["messages"][0].content
    assert "[1] (source: policy.pdf)" in system_prompt


def test_rag_without_hits_returns_none(fake_providers, monkeypatch):
    async def no_hits(*a, **kw):
        return []

    monkeypatch.setattr(rag_workflow, "search", no_hits)
    assert run(rag_workflow.run_rag("anything", user_id="u1")) is None


def test_retrieval_query_expands_short_follow_ups():
    history = [Message(role="user", content="refund policy for electronics"),
               Message(role="assistant", content="...")]
    assert "electronics" in rag_workflow.retrieval_query("and for books?", history)
    long_q = "what is the warranty period for laptops bought online last year"
    assert rag_workflow.retrieval_query(long_q, history) == long_q


def test_search_always_filters_by_owner(fake_providers, monkeypatch):
    from app.ai.db import vector_store

    captured = {}

    class FakeCollection:
        def query(self, **kw):
            captured.update(kw)
            return {"ids": [["a:0"]], "documents": [["text"]],
                    "metadatas": [[{"doc_id": "a", "owner": "u1"}]], "distances": [[0.2]]}

    monkeypatch.setattr(vector_store, "_collection", lambda ns: FakeCollection())
    hits = run(vector_store.search("documents", "q", user_id="u1", k=3))
    assert captured["where"] == {"owner": {"$in": ["u1", vector_store.SHARED_OWNER]}}
    assert hits[0].score == pytest.approx(0.8)
    with pytest.raises(ValueError):
        run(vector_store.search("documents", "q", user_id=""))


# ---- LangGraph chat orchestration ----------------------------------------- #
@pytest.fixture
def graph_env(monkeypatch):
    pytest.importorskip("langgraph")
    from app.ai.orchestration import graph
    from app.ai.orchestration.router import RouteDecision
    from app.ai.workflows.qa_workflow import QAResult

    saved: list[dict] = []
    env = {"route": RouteDecision("general", 0.5, "x"), "hits": [], "qa": None}

    async def ensure_session(user_id, session_id=None):
        return session_id or "s1"

    async def get_history(sid, uid, limit=20):
        return []

    async def tables():
        return ["orders"]

    async def add_exchange(sid, uid, user_text, answer, meta=None):
        saved.append({"answer": answer, "meta": meta})

    async def route_message(message, tables, provider=None):
        return env["route"]

    async def retrieve(question, *, user_id, history=None, k=None):
        return env["hits"]

    async def run_qa(question, *, history=None, provider=None):
        if env["qa"] is None:
            raise CannotAnswer("no")
        return env["qa"]

    async def recall(*a, **kw):
        return []

    for name, fn in [("ensure_session", ensure_session), ("get_history", get_history),
                     ("get_table_names", tables), ("add_exchange", add_exchange),
                     ("route_message", route_message), ("retrieve", retrieve),
                     ("run_qa", run_qa), ("recall", recall)]:
        monkeypatch.setattr(graph, name, fn)
    env["saved"], env["graph"], env["QAResult"] = saved, graph.build_chat_graph(), QAResult
    return env


def _invoke(env, message="hello there"):
    return run(env["graph"].ainvoke({"message": message, "user_id": "u1"}))


def test_graph_sql_route(fake_providers, graph_env):
    from app.ai.orchestration.router import RouteDecision

    graph_env["route"] = RouteDecision("data_query", 0.9, "table")
    graph_env["qa"] = graph_env["QAResult"](question="q", sql="SELECT 1", answer="42 orders")
    state = _invoke(graph_env)
    assert state["answered_by"] == "data_query" and state["answer"] == "42 orders"
    assert graph_env["saved"][0]["meta"]["sql"] == "SELECT 1"


def test_graph_sql_falls_back_to_documents(fake_providers, graph_env):
    from app.ai.orchestration.router import RouteDecision

    groq, _ = fake_providers
    groq.replies = ["From the docs [1]."]
    graph_env["route"] = RouteDecision("data_query", 0.6, "keywords")
    graph_env["hits"] = [_hit("some passage", 0.9)]
    state = _invoke(graph_env)
    assert state["answered_by"] == "knowledge" and state["sources"][0]["source"] == "policy.pdf"


def test_graph_knowledge_without_hits_uses_chat(fake_providers, graph_env):
    from app.ai.orchestration.router import RouteDecision

    groq, _ = fake_providers
    groq.replies = ["plain reply"]
    graph_env["route"] = RouteDecision("knowledge", 0.8, "doc words")
    state = _invoke(graph_env)
    assert state["answered_by"] == "general" and state["answer"] == "plain reply"


def test_graph_general_needs_strong_hits(fake_providers, graph_env):
    groq, _ = fake_providers
    groq.replies = ["chat reply"]
    graph_env["hits"] = [_hit("weakly related", 0.40)]  # below RAG_MIN_SCORE + bonus
    assert _invoke(graph_env)["answered_by"] == "general"


# ---- small utilities ------------------------------------------------------- #
def test_context_builder_keeps_budget_and_user_first():
    history = [Message(role="assistant", content="orphan")] + [
        Message(role="user" if i % 2 == 0 else "assistant", content="x" * 2000)
        for i in range(20)
    ]
    ctx = build_context(system_prompt="sys", user_message="question", history=history,
                        sections=[("Memories", "- likes tea")], max_tokens=3000)
    assert ctx.token_estimate <= 2000 + 50
    assert ctx.history and ctx.history[0].role == "user"
    assert "likes tea" in ctx.system and ctx.dropped_messages > 0


def test_prompt_render_fails_on_missing_value():
    template = load_prompt("sql_agent")
    with pytest.raises(KeyError):
        render(template, TODAY="2026-01-01")
    assert "{\"sql\"" in render(template, TODAY="t", MAX_ROWS=5, SCHEMA="s")


def test_rate_limiter():
    limiter = RateLimiter(per_minute=60, burst=2)
    limiter.check("u")
    limiter.check("u")
    with pytest.raises(RateLimitExceeded):
        limiter.check("u")
    limiter.check("other-user")  # separate bucket


def test_chunking_splits_with_overlap():
    pytest.importorskip("langchain_text_splitters")
    from app.ai.utils.chunking import split_text

    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(20))
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 5 and all(len(c) <= 500 for c in chunks)


# ---- real ChromaDB (in-memory) ---------------------------------------------- #
def test_chroma_end_to_end_isolation(fake_providers):
    chromadb = pytest.importorskip("chromadb")
    from app.ai.db import vector_store
    from app.ai.db.vector_store import SHARED_OWNER, VectorItem

    vector_store.set_client(chromadb.EphemeralClient())
    try:
        items = [
            VectorItem(content="alpha secret of user one", owner="u1", doc_id="d1"),
            VectorItem(content="beta note of user two", owner="u2", doc_id="d2"),
            VectorItem(content="gamma shared handbook", owner=SHARED_OWNER, doc_id="d3",
                       metadata={"filename": "handbook.md", "extra": None}),
        ]
        assert run(vector_store.add_documents("documents", items)) == 3

        hits = run(vector_store.search("documents", "anything", user_id="u1", k=10))
        assert {h.doc_id for h in hits} == {"d1", "d3"}  # never u2's chunk
        assert all(-1.0 <= h.score <= 1.0 for h in hits)

        docs = run(vector_store.list_documents("documents", owner="u1"))
        assert [d["doc_id"] for d in docs] == ["d1"]
        assert run(vector_store.delete_documents("documents", owner="u2", doc_id="d1")) == 0
        assert run(vector_store.delete_documents("documents", owner="u1", doc_id="d1")) == 1
        assert run(vector_store.search("documents", "x", user_id="u1", k=10,
                                       include_shared=False)) == []
    finally:
        vector_store.close_vector_store()
