# VaakDB backend

FastAPI backend for VaakDB, an AI database interface. From the chat, a user can:

- list schemas and tables and describe a table (no LLM involved)
- ask read questions in natural language (the existing `app/ai` text-to-SQL path)
- store a table from an uploaded **PDF / CSV / XLSX** into a schema
- create tables and insert, update or delete rows

Every write goes through a **preview** first and only runs after an explicit confirm.

## Architecture

```
meta_db   (META_DATABASE_URL)  users, conversations, uploads, operations,
                               audit_logs (append-only), query_history
target_db (TARGET_DB_RO_URL)   reads    as vaak_ro (read-only, short timeout)
          (TARGET_DB_RW_URL)   writes   as vaak_rw (creates/changes workspace schemas only)
```

```
POST /chat ─► route_intent ─┬─ list/describe ─► metadata_ops ───────────────────────────► respond
                            ├─ select_query ──► app/ai text-to-SQL (read-only) ─────────► respond
                            ├─ ingest/create/insert/update/delete
                            │      ─► build_plan (LLM → Pydantic plan, never SQL)
                            │      ─► validate_plan ─(repairable, <2 tries)─► repair_plan ─┐
                            │             └─(ok)─► pending operation + preview ─► respond   │
                            │             ▲───────────────────────────────────────────────┘
                            └─ clarify / unsupported ───────────────────────────────────► respond
POST /operations/{id}/confirm ─► re-validate (own workspace, live DB) ─► plan_executor (1 transaction)
```

| Piece | File |
|---|---|
| Settings, errors (stable `code`s), log scrubbing | `app/core/` |
| Models + migration | `app/db/models.py`, `migrations/versions/b3f1c2d4e5a6_*.py` |
| Target engines (RO/RW), URL encoding | `app/db/target_db.py`, `app/db/urls.py` |
| Identifier rules + type whitelist / value casting | `app/sql/parser.py`, `app/sql/validator.py` |
| Personal workspaces (isolation), destructive-op policy | `app/security/authorization.py`, `app/sql/policy.py` |
| Natural-language SELECT inside the workspace | `app/services/nl_query.py` |
| Plan models | `app/schemas/plans.py` |
| Validator / executor / pending operations | `app/services/plan_validator.py`, `plan_executor.py`, `operations_service.py` |
| File ingestion | `app/services/ingest/` |
| Metadata ops | `app/services/metadata_ops.py` |
| LangGraph orchestration | `app/services/chat_service.py` |
| LLM: structured output, intent router, planner | `app/ai/dbops/` (new; existing `app/ai` untouched) |
| Postgres roles | `scripts/create_ro_role.sql`, `scripts/create_rw_role.sql`, `scripts/setup_roles.py` |

## Setup

```bash
pip install -r requirements.txt -r app/ai/requirements.txt     # dev: -r requirements-dev.txt
cp .env.example .env                                            # fill in URLs, JWT_SECRET, LLM keys

# databases: meta_db + target_db must exist (docker compose creates both via infra/init.sql)
alembic upgrade head                                            # meta_db tables
python scripts/setup_roles.py --admin-url "postgresql://postgres:PASSWORD@localhost:5432/target_db"
#   -> creates vaak_ro, vaak_rw; prints TARGET_DB_RO_URL / TARGET_DB_RW_URL for .env

python scripts/create_user.py you@example.com                  # or register in the app
python main.py                                                  # http://127.0.0.1:8000/docs
```

### Personal workspaces (user-level isolation)

There are no roles. Every account can do everything (list, describe, query, export, create,
insert, update, delete, replace) **on its own data** and nothing on anyone else's:

- **target_db**: each user's schemas are stored as `u<user_id>_<name>`. Users only see and type the
  logical name (`college`); the backend maps it on every request, so another user's schema can't
  even be named. Schemas are created automatically with the first table (default: `DEFAULT_SCHEMA`,
  `main`), up to `MAX_SCHEMAS_PER_USER`.
- **Natural-language SELECT** sees only the tables of one of the user's schemas; the SQL guard
  rejects anything else and every table reference is rewritten to the user's physical schema.
- **meta_db** (uploads, operations, conversations, query history, projects) is filtered by `user_id`
  in every query; another user's id gives 404.
- The original `/api/v1/ai/*` routes read one shared schema, so they are off unless
  `ENABLE_AI_ROUTES=true`.

## API walkthrough (curl)

```bash
API=http://127.0.0.1:8000/api/v1
TOKEN=$(curl -s $API/auth/login -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"..."}' | jq -r .access_token)
AUTH="Authorization: Bearer $TOKEN"
```

**1. Store a table from a PDF**

```bash
# upload: type checked by magic bytes, max MAX_UPLOAD_MB, stored under a random name
UPLOAD=$(curl -s $API/files/upload -H "$AUTH" -F "file=@tests/fixtures/students.pdf" | jq -r .upload_id)

# chat: returns an operation_preview (DDL, column mapping, first 10 rows, failed rows, expiry)
curl -s "$API/chat?stream=false" -H "$AUTH" -H 'Content-Type: application/json' \
  -d "{\"message\":\"store this as students in schema college\",\"upload_id\":\"$UPLOAD\"}" | jq
OP=<data.operation_id from the preview>

curl -s $API/operations/$OP -H "$AUTH" | jq          # status + preview
curl -s -X POST $API/operations/$OP/confirm -H "$AUTH" | jq   # executes; repeat = same result, no re-run
curl -s -X POST $API/operations/$OP/cancel -H "$AUTH" | jq    # (instead of confirm)
```

**2. Streaming chat (SSE)**: `status` events per step, then `result` or `error`

```bash
curl -N $API/chat -H "$AUTH" -H 'Accept: text/event-stream' -H 'Content-Type: application/json' \
  -d '{"message":"show tables in schema college"}'
# event: status  data: {"stage":"route_intent","message":"Understanding your request"}
# event: result  data: {"type":"table_list","data":{...},"message":"Schema college has 1 table(s): students.","conversation_id":"..."}
```

**3. Metadata, rows, export**

```bash
curl -s $API/schemas -H "$AUTH"
curl -s $API/schemas/college/tables -H "$AUTH"
curl -s $API/schemas/college/tables/students -H "$AUTH"
curl -s "$API/schemas/college/tables/students/rows?page=1&page_size=50&sort=gpa&dir=desc" -H "$AUTH"
curl -s $API/schemas/college/tables/students/csv -H "$AUTH" -o students.csv
```

**4. CRUD through chat** (same preview → confirm flow)

```bash
chat() { curl -s "$API/chat?stream=false" -H "$AUTH" -H 'Content-Type: application/json' -d "{\"message\":\"$1\"}" | jq; }
chat "create a table college.courses with id integer primary key and title text"
chat "add a student roll 106 named Zoya aged 22 to college.students"
chat "set gpa to 9.0 for roll_no 101 in college.students"        # preview shows affected row count
chat "delete students with gpa below 7 from college.students"
chat "how many students are older than 20?"                       # read → existing text-to-SQL
```

### Response envelope (`/chat`)

`{"type": ..., "data": ..., "message": "...", "conversation_id": "..."}` where `type` is one of
`schema_list | table_list | table_description | query_result | operation_preview | operation_result | message | error`
(see `app/schemas/result.py`). Errors everywhere are `{"error": {"code": "TABLE_EXISTS", "message": "..."}}`
(in chat: `{"type": "error", "data": {"code", "message"}}`) and never contain SQL, stack traces or keys.

## Safety model

- The LLM only fills Pydantic plans (`app/schemas/plans.py`); the backend builds every statement with SQLAlchemy Core,
  identifiers checked against `^[a-z_][a-z0-9_]{0,62}$`, values as bound parameters.
- Rows from files are never generated by the LLM: it sees the header + first 30 rows and returns a column mapping;
  the rows come from the extracted file.
- File text is untrusted: the router never sees it, the planner sees it only inside `<file_data>` blocks, and the
  target schema/table and `append`/`replace` mode come from the user's message.
- Writes: pending operation → confirm by the same user within `OPERATION_TTL_MINUTES` → re-validation with the user's
  own workspace → one transaction (DDL + batched inserts of 500), `statement_timeout`, `MAX_INSERT_ROWS` → rollback on
  any error. A killed process leaves nothing behind (tested).
- UPDATE/DELETE need a WHERE (or an explicit "all rows" from the user), and the preview shows the affected row count.
- Everything is written to `audit_logs`, which a trigger keeps append-only.

## Tests

```bash
pip install -r requirements-dev.txt
# needs PostgreSQL: either Docker running (testcontainers) or any server:
TEST_PG_SUPERUSER_URL="postgresql://postgres:PASSWORD@localhost:5432/postgres" pytest
ruff check . && mypy app scripts
```

The suite creates throwaway databases and roles (`vaak_test_*`) and drops them afterwards; it never touches the
databases in `.env`. Fixture PDFs are generated by `tests/fixtures/make_fixtures.py`.

### Limits / not done yet

- OCR for scanned PDFs (returns `SCANNED_PDF_UNSUPPORTED`).
- Natural-language SELECT answers from one schema of the workspace at a time (no cross-schema joins).
- DROP TABLE / ALTER / TRUNCATE are not exposed through chat (replacing a table on import is the only drop).
