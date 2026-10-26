"""Routing: heuristics, the LLM classifier (faked), its failure fallback, and
provider preference."""
from __future__ import annotations

import asyncio

import pytest

from app.ai.orchestration.router import choose_provider, heuristic_route, route_message

TABLES = ["orders", "customers", "order_items"]


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("msg", ["hi", "Hello!", "thanks", "good morning", "what can you do?"])
def test_greetings_are_general(msg):
    d = heuristic_route(msg, TABLES)
    assert d.route == "general" and d.confidence >= 0.9


@pytest.mark.parametrize("msg", [
    "How many orders did we get last week?",
    "list customers from Delhi",
    "show me order items for order 12",
])
def test_table_mentions_are_data_queries(msg):
    d = heuristic_route(msg, TABLES)
    assert d.route == "data_query" and d.confidence >= 0.9


@pytest.mark.parametrize("msg", [
    "What does the refund policy document say about returns?",
    "summarize the pdf I uploaded",
    "according to the contract, when is payment due?",
])
def test_document_questions_are_knowledge(msg):
    assert heuristic_route(msg, TABLES).route == "knowledge"


def test_no_tables_never_routes_to_sql():
    assert heuristic_route("how many sales this month", []).route == "general"


def test_confident_rule_skips_llm(fake_providers):
    groq, _ = fake_providers
    d = run(route_message("how many orders today", TABLES))
    assert d.route == "data_query" and groq.calls == []


def test_llm_decides_when_unsure(fake_providers):
    groq, _ = fake_providers
    groq.replies = ['{"route": "data_query", "confidence": 0.8}']
    d = run(route_message("what was our best month?", TABLES))
    assert d.route == "data_query" and d.reason == "llm"
    assert groq.calls[0]["json_mode"] is True
    assert "orders" in groq.calls[0]["messages"][0].content  # tables are in the prompt


def test_llm_garbage_falls_back_to_heuristic(fake_providers):
    groq, gemini = fake_providers
    groq.replies = ["not json at all"]
    d = run(route_message("tell me about the weather in general", TABLES))
    assert d.route == "general" and d.reason != "llm"


def test_llm_invalid_route_falls_back(fake_providers):
    groq, _ = fake_providers
    groq.replies = ['{"route": "delete_everything"}']
    assert run(route_message("tell me something", TABLES)).reason != "llm"


def test_llm_data_route_without_tables_becomes_knowledge(fake_providers):
    groq, _ = fake_providers
    groq.replies = ['{"route": "data_query", "confidence": 0.9}']
    assert run(route_message("what were total sales?", [])).route == "knowledge"


def test_choose_provider():
    assert choose_provider("sql") == "groq"
    assert choose_provider("file") == "gemini"
    assert choose_provider("chat", has_attachments=True) == "gemini"
    assert choose_provider("chat", text_len=10_000_000) == "gemini"
    assert choose_provider("chat", override="gemini") == "gemini"
