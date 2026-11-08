"""LLM pieces of the chat -> DB operations flow (P1).

- llm.py      get_chat_model() / .with_structured_output(Model)
- prompts.py  router / ingest / plan prompts (file text is untrusted data)
- intent.py   route_intent(): message -> IntentDecision
- planner.py  draft_ingest() / draft_plan(): request -> structured plan (never SQL)

Orchestration (LangGraph), validation and execution live in the backend
(app/services/chat_service.py, plan_validator.py, plan_executor.py).
"""
