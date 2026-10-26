"""VaakDB AI module: natural-language access to PostgreSQL plus RAG over
user-uploaded documents and indexed tables, orchestrated with LangGraph.

Kept import-light on purpose (importing app.ai.config must not pull in every
agent). Entry points:
    app.ai.api.routes.router        FastAPI router (/ai/...)
    app.ai.lifecycle.lifespan       startup/shutdown hooks
    app.ai.orchestration.orchestrator.orchestrator
"""
