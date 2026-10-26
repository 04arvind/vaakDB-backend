# VaakDB AI module (`app/ai`)

Natural-language access to PostgreSQL **plus** RAG over user-uploaded documents,
orchestrated with **LangGraph**, using **Groq** (fast text) and **Gemini**
(multimodal, long context, embeddings), with **ChromaDB** as the vector store.

```
                         ┌──────────── /ai/chat (LangGraph) ─────────────┐
 user message ─► route ─►│ data_query ─► SQL agent ─► sql_guard ─► Postgres (read-only role)
                         │ knowledge  ─► Chroma search (user docs + indexed rows) ─► RAG answer [1][2]
                         │ general    ─► chat (+ long-term memories)            │
                         └─────────────────────► persist (history, memory) ──────┘

 uploads ─► /ai/documents ─► Gemini transcribe (PDF/image/audio) ─► chunk (LangChain) ─► embed ─► Chroma
 tables  ─► index_workflow.index_table ─► rows as text ─► Chroma ("database" namespace)
 text/files ─► /ai/ingest ─► extractor ─► Pydantic validation ─► INSERT into registered tables only
```

## Layout

| Folder | What lives there |
|---|---|
| `providers/` | Groq + Gemini over raw HTTP (httpx): retries, error mapping, embeddings |
| `orchestration/` | `graph.py` (LangGraph), `router.py`, `fallback.py` (provider failover), `orchestrator.py` |
| `agents/` | SQL, RAG, chat, extractor, file reader, summarizer, memory agents |
| `workflows/` | QA (text-to-SQL with repair loop), RAG, indexing, structured ingest, file ingest |
| `db/` | engines (read-only + write roles), `sql_guard.py`, reader, writer, schema inspector, Chroma vector store |
| `memory/` | conversation history, long-term memory, context budgeting |
| `tools/` | function-calling tools wrapping the guarded DB layer |
| `api/` | FastAPI router, auth / rate-limit / request-id dependencies |
| `prompts/system/*.md` | all system prompts |
| `test/` | `pytest app/ai/test` (no network, DB or Chroma server needed) |

## Setup

```bash
pip install -r requirements.txt -r app/ai/requirements.txt
```

Mount it in the FastAPI app:

```python
from fastapi import FastAPI
from app.ai.api.routes import router as ai_router
from app.ai.lifecycle import lifespan

app = FastAPI(lifespan=lifespan)
app.include_router(ai_router)
```

If the app has its own auth, replace the JWT dependency:
`app.dependency_overrides[app.ai.api.deps.current_user_id] = your_dependency` (it must return the user id).

### Environment (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `GROQ_API_KEY`, `GEMINI_API_KEY` | – | at least one; Gemini is required for embeddings/RAG and file reading |
| `DATABASE_READ_URL` | – | read-only role, used for LLM-written SQL |
| `DATABASE_WRITE_URL` | – | restricted role for history, memories and ingest |
| `AI_JWT_SECRET` / `AI_ALLOW_ANONYMOUS` | – / `false` | anonymous is refused when `AI_ENV=production` |
| `CHROMA_MODE` | `persistent` | `persistent` (`CHROMA_PATH`), `http` (`CHROMA_HOST`/`PORT`), `memory` |
| `RAG_TOP_K`, `RAG_MIN_SCORE` | `6`, `0.35` | retrieval breadth / relevance floor (cosine) |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | `1200`, `150` | characters |
| `RAG_INDEX_TABLES` | – | tables indexed by `python -m app.ai.workflows.index_workflow` |
| `AI_READABLE_TABLES` / `AI_HIDDEN_TABLES` | – | allow-list / deny-list for the SQL agent |
| `SQL_MAX_ROWS`, `SQL_TIMEOUT_MS`, `SQL_MAX_REPAIRS` | `100`, `5000`, `2` | |
| `RATE_LIMIT_PER_MINUTE`, `RATE_LIMIT_BURST` | `30`, `10` | per user, per process |
| `AI_AUTO_CREATE_TABLES` | `false` | create `ai_conversations`/`ai_messages`/`ai_memories` at startup |

All settings: `config.py`.

### PostgreSQL roles (the real security boundary)

```sql
-- read-only role for LLM-generated SQL
CREATE ROLE ai_reader LOGIN PASSWORD '...';
GRANT USAGE ON SCHEMA public TO ai_reader;
GRANT SELECT ON orders, customers, products TO ai_reader;          -- only what the AI may read
GRANT SELECT (id, name, email) ON users TO ai_reader;              -- column-level: no password_hash
ALTER ROLE ai_reader SET default_transaction_read_only = on;

-- write role: chat history, memories, and registered ingest targets only
CREATE ROLE ai_writer LOGIN PASSWORD '...';
GRANT USAGE ON SCHEMA public TO ai_writer;
GRANT SELECT, INSERT, UPDATE, DELETE ON ai_conversations, ai_messages, ai_memories TO ai_writer;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO ai_writer;
```

`sql_guard.py` is defence in depth on top of this: single SELECT only, table
allow-list, blocked system functions, no sensitive or whole-row column access,
enforced LIMIT.

## Endpoints (prefix `/ai`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | LangGraph-routed conversation (SQL / RAG / chat) |
| POST | `/query` | always text-to-SQL; `include_rows` returns the rows |
| POST | `/ask` | always RAG over the user's knowledge base, with sources |
| POST | `/documents` | upload PDF / image / audio / text into the knowledge base |
| POST | `/documents/text` | add pasted text to the knowledge base |
| GET / DELETE | `/documents`, `/documents/{doc_id}` | list / remove documents |
| POST | `/ingest`, `/ingest/file` | extract records into registered tables (`dry_run` to preview) |
| GET / DELETE | `/sessions`, `/sessions/{id}` | conversation history |
| GET / DELETE | `/memories`, `/memories/{id}` | long-term memory (user-visible and deletable) |
| GET | `/health`, `/ready` | liveness / readiness (providers + both DB connections) |

Structured ingest only writes to tables registered in `db/targets.py`.

## Tests

```bash
pytest app/ai/test
```
