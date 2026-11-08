"""Chat -> DB operations orchestration (LangGraph).

    START -> route_intent
      +- list_schemas / list_tables / describe_table -> metadata_op -> respond -> END
      +- select_query -> select_query (existing text-to-SQL, app.ai) -> respond
      +- ingest_file / create_table / insert / update / delete
      |     -> build_plan (LLM, structured) -> validate_plan
      |          validate_plan -(repairable error, attempts < 2)-> repair_plan -> validate_plan
      |          validate_plan -(ok)-> create_pending_operation -> respond
      +- clarify / unsupported -> respond

Confirmation is NOT a node: POST /operations/{id}/confirm calls plan_executor.

Everything runs inside the caller's personal workspace: schema names in plans
are logical and are mapped to the caller's private schemas by the validator,
metadata ops and the workspace query runner. File content never reaches
route_intent, and the user's own message decides the target table and mode
(see _enforce_user_intent).
"""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.ai.dbops.intent import route_intent as llm_route_intent
from app.ai.dbops.llm import ChatModel, LLMBadOutputError, LLMUnavailableError, get_chat_model
from app.ai.dbops.planner import draft_ingest, draft_plan
from app.core.config import settings
from app.core.exceptions import AppError, ErrorCode
from app.db.session import SessionLocal
from app.db.target_db import get_read_engine
from app.repositories import query_runs
from app.schemas.plans import (
    METADATA_INTENTS,
    WRITE_INTENTS,
    AnyPlan,
    ColumnDef,
    CreateTablePlan,
    DeleteRowsPlan,
    IngestDraft,
    Intent,
    IntentDecision,
    UpdateRowsPlan,
)
from app.security.authorization import Caller, to_logical, to_physical
from app.services import metadata_ops, operations_service, schema_service
from app.services.audit_service import audit
from app.services.ingest import Extraction
from app.services.nl_query import WorkspaceQueryResult, run_workspace_query
from app.services.plan_validator import PlanValidationError, ValidatedPlan, validate_plan
from app.services.upload_service import load_upload
from app.sql.parser import normalize_identifier

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 2  # plan attempts before giving up (first try + repairs)
MAX_FILE_TABLES_FOR_LLM = 5

UNSUPPORTED_MESSAGE = (
    "I can list schemas and tables, describe a table, answer questions about your data, "
    "store a table from an uploaded PDF/CSV/XLSX, and create tables or insert, update and "
    "delete rows (always with a preview you confirm first). Dropping, truncating or altering "
    "tables is not available through chat."
)
STATUS_TEXT = {
    "route_intent": "Understanding your request",
    "metadata_op": "Reading the database catalog",
    "select_query": "Querying your data",
    "build_plan": "Preparing a plan",
    "validate_plan": "Checking the plan",
    "repair_plan": "Fixing the plan",
    "create_pending_operation": "Creating a preview",
    "respond": "Done",
}
_APPEND_WORDS = re.compile(
    r"\b(append|add (?:them |it |the rows )?to (?:the )?existing|existing table)\b", re.I
)
_REPLACE_WORDS = re.compile(r"\b(replace|overwrite|recreate|re-create)\b", re.I)
_ALL_ROWS_WORDS = re.compile(r"\b(all|every|entire)\b", re.I)


SelectRunner = Callable[[Caller, str, str], Awaitable[WorkspaceQueryResult]]


@dataclass
class ChatDeps:
    chat_model_factory: Callable[[], ChatModel] = field(
        default_factory=lambda: lambda: get_chat_model(settings.llm_provider)
    )
    run_select: SelectRunner = run_workspace_query
    record_history: bool = True


class ChatState(TypedDict, total=False):
    user: Caller
    message: str
    upload_id: str | None
    conversation_id: str | None
    intent: str
    decision: IntentDecision
    schema: str | None
    table: str | None
    plan: Any
    draft: Any
    validated: ValidatedPlan
    error: dict[str, Any] | None
    attempts: int
    operation_id: str | None
    sql: str | None
    response: dict[str, Any]


# --------------------------------------------------------------------------- #
# envelopes
# --------------------------------------------------------------------------- #
def error_envelope(exc: AppError) -> dict[str, Any]:
    return {"type": "error", "data": exc.to_dict(), "message": exc.message}


def message_envelope(text: str, **data: Any) -> dict[str, Any]:
    return {"type": "message", "data": data or None, "message": text}


def _err(state: ChatState, exc: AppError) -> ChatState:
    if exc.code in (ErrorCode.FORBIDDEN_SCHEMA, ErrorCode.DESTRUCTIVE_OPERATION_BLOCKED):
        audit("denied", user_id=state["user"].user_id, status="denied",
              operation_type=state.get("intent"), target_schema=state.get("schema"),
              target_table=state.get("table"), error=f"{exc.code}: {exc.message}")
    return {"response": error_envelope(exc)}


def _llm_error(exc: Exception) -> AppError:
    if isinstance(exc, LLMUnavailableError):
        return AppError(ErrorCode.LLM_UNAVAILABLE)
    return AppError(ErrorCode.LLM_BAD_OUTPUT)


def _norm(value: str | None, kind: str) -> str | None:
    if value is None or not str(value).strip():
        return None
    return normalize_identifier(value, kind)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# helpers that touch the databases (sync, run in threads)
# --------------------------------------------------------------------------- #
def _load_upload_sync(caller: Caller, upload_id: str) -> tuple[uuid.UUID, Extraction]:
    with SessionLocal() as db:
        upload, extraction = load_upload(db, caller, upload_id)
        return upload.id, extraction


def _user_schemas_sync(caller: Caller) -> list[str]:
    """Logical names of the caller's schemas."""
    with get_read_engine().connect() as conn:
        physical = schema_service.owned_schemas(conn, caller.prefix)
    return [name for p in physical if (name := to_logical(caller, p))]


def _table_description_sync(caller: Caller, schema: str, table: str) -> str | None:
    with get_read_engine().connect() as conn:
        info = schema_service.get_table(conn, to_physical(caller, schema), table)
    if info is None:
        return None
    lines = []
    for c in info.columns:
        flags = [f for f, on in (("primary key", c.primary_key), ("not null", not c.nullable),
                                 ("has default", c.default is not None)) if on]
        lines.append(f"- {c.name}: {c.display_type}" + (f" ({', '.join(flags)})" if flags else ""))
    return f"TABLE {schema}.{table}\n" + "\n".join(lines)


def _validate_sync(plan: AnyPlan, caller: Caller) -> ValidatedPlan:
    with get_read_engine().connect() as conn:
        return validate_plan(plan, caller=caller, conn=conn)


def _create_pending_sync(caller: Caller, validated: ValidatedPlan, conversation_id: str | None,
                         upload_id: str | None) -> dict[str, Any]:
    with SessionLocal() as db:
        op = operations_service.create_pending(
            db, caller=caller, validated=validated,
            conversation_id=uuid.UUID(conversation_id) if conversation_id else None,
            upload_id=uuid.UUID(upload_id) if upload_id else None,
        )
        return dict(op.preview_json)


# --------------------------------------------------------------------------- #
# plan shaping
# --------------------------------------------------------------------------- #
def ingest_to_plan(draft: IngestDraft, extraction: Extraction, upload_id: uuid.UUID) -> CreateTablePlan:
    """Column mapping from the LLM + rows straight from the extracted file."""
    if draft.table_index >= len(extraction.tables):
        raise PlanValidationError(ErrorCode.PLAN_INVALID,
                                  f"table_index {draft.table_index} does not exist; the file has "
                                  f"{len(extraction.tables)} table(s).", repairable=True)
    source = extraction.tables[draft.table_index]
    bad = [c.source_index for c in draft.columns if c.source_index >= source.width]
    if bad:
        raise PlanValidationError(ErrorCode.PLAN_INVALID,
                                  f"source_index {bad} out of range: the table has {source.width} "
                                  "columns (0-based).", repairable=True)
    columns = [ColumnDef(name=c.name, pg_type=c.pg_type, nullable=c.nullable,
                         primary_key=c.primary_key, source=source.header[c.source_index])
               for c in draft.columns]
    rows = [[row[c.source_index] for c in draft.columns] for row in source.rows]
    return CreateTablePlan(schema_name=draft.schema_name, table=draft.table, columns=columns, rows=rows,
                           if_exists=draft.if_exists, upload_id=upload_id,
                           source_table_index=draft.table_index)


def _enforce_user_intent(plan: Any, message: str, schema: str, table: str) -> Any:
    """The user's message, not the model (or a file), decides target and mode."""
    update: dict[str, Any] = {}
    if plan.schema_name != schema or plan.table != table:
        log.info("Plan target %s.%s overridden by the user's %s.%s",
                 plan.schema_name, plan.table, schema, table)
        update.update(schema_name=schema, table=table)
    mode = getattr(plan, "if_exists", None)
    if mode == "append" and not _APPEND_WORDS.search(message):
        update["if_exists"] = "fail"
    if mode == "replace" and not _REPLACE_WORDS.search(message):
        update["if_exists"] = "fail"
    if isinstance(plan, (UpdateRowsPlan, DeleteRowsPlan)) and plan.all_rows \
            and not _ALL_ROWS_WORDS.search(message):
        update["all_rows"] = False
    return plan.model_copy(update=update) if update else plan


def preview_message(preview: dict[str, Any]) -> str:
    target = f"{preview['target']['schema']}.{preview['target']['table']}"
    kind = preview["operation_type"]
    if kind == "create_table":
        if preview.get("mode") == "append" and preview.get("table_exists"):
            text = f"Ready to append {preview['rows_to_write']} row(s) to {target}."
        else:
            verb = "replace" if preview.get("table_exists") else "create"
            text = (f"Ready to {verb} {target} with {len(preview['columns'])} column(s)"
                    + (f" and insert {preview['rows_to_write']} row(s)." if preview.get("total_rows")
                       else "."))
    elif kind == "insert_rows":
        text = f"Ready to insert {preview['rows_to_write']} row(s) into {target}."
    elif kind == "update_rows":
        text = f"Ready to update {preview['affected_rows']} row(s) in {target}."
    else:
        text = f"Ready to delete {preview['affected_rows']} row(s) from {target}."
    if preview.get("failed_row_count"):
        text += f" {preview['failed_row_count']} row(s) don't match the column types and will be skipped."
    return text + (f" Review the preview and confirm within {settings.operation_ttl_minutes} minutes.")


# --------------------------------------------------------------------------- #
# graph
# --------------------------------------------------------------------------- #
def build_chat_graph(deps: ChatDeps | None = None) -> Any:
    deps = deps or ChatDeps()

    async def route_intent(state: ChatState) -> ChatState:
        caller = state["user"]
        upload_id = state.get("upload_id")
        try:
            schemas = await asyncio.to_thread(_user_schemas_sync, caller)
        except Exception:
            schemas = []
        try:
            decision = await llm_route_intent(
                state["message"], chat_model=deps.chat_model_factory(),
                has_upload=bool(upload_id), schemas=schemas or [settings.default_schema],
            )
        except (LLMUnavailableError, LLMBadOutputError) as exc:
            return {"response": error_envelope(_llm_error(exc))}

        intent = decision.intent
        if upload_id and intent in (Intent.create_table, Intent.insert_rows, Intent.clarify):
            intent = Intent.ingest_file  # an attached file wins
        out: ChatState = {"intent": intent.value, "decision": decision, "attempts": 0}

        raw_schema, raw_table = decision.schema_name, decision.table
        if raw_table and "." in raw_table and not raw_schema:
            raw_schema, raw_table = raw_table.split(".", 1)
        try:
            schema, table = _norm(raw_schema, "schema"), _norm(raw_table, "table")
            if schema:
                to_physical(caller, schema)  # rejects system schemas / over-long names early
        except AppError as exc:
            return {**out, **_err(state, exc)}
        out["schema"], out["table"] = schema, table

        if intent == Intent.clarify:
            question = decision.clarification_question or "Could you tell me a bit more about what you need?"
            return {**out, "response": message_envelope(question, clarification=True)}
        if intent == Intent.unsupported:
            return {**out, "response": message_envelope(UNSUPPORTED_MESSAGE)}

        if intent in WRITE_INTENTS:
            if intent == Intent.ingest_file and not upload_id:
                return {**out, "response": message_envelope(
                    "Please attach the file (PDF, CSV or XLSX) you want to store, then ask again.",
                    clarification=True)}
            if not table:
                question = decision.clarification_question or "What should the table be called?"
                return {**out, "response": message_envelope(question, clarification=True)}
            if not schema:  # writes go to the default schema of the workspace
                out["schema"] = schemas[0] if len(schemas) == 1 else settings.default_schema
            return out

        if intent in (Intent.list_tables, Intent.describe_table, Intent.select_query) and not schema:
            if len(schemas) == 1:
                out["schema"] = schemas[0]
            elif not schemas:
                return {**out, "response": message_envelope(
                    "Your workspace is empty. Create a table or attach a file to store one.")}
            elif settings.default_schema in schemas and intent != Intent.list_tables:
                out["schema"] = settings.default_schema
            else:
                return {**out, "response": message_envelope(
                    f"Which schema? You have: {', '.join(schemas)}.", clarification=True)}
        if intent == Intent.describe_table and not table:
            return {**out, "response": message_envelope("Which table should I describe?",
                                                         clarification=True)}
        return out

    async def metadata_op(state: ChatState) -> ChatState:
        caller, intent = state["user"], Intent(state["intent"])
        try:
            if intent == Intent.list_schemas:
                env = await asyncio.to_thread(metadata_ops.list_schemas, caller)
            elif intent == Intent.list_tables:
                env = await asyncio.to_thread(metadata_ops.list_tables, caller, state["schema"] or "")
            else:
                env = await asyncio.to_thread(metadata_ops.describe_table, caller,
                                              state["schema"] or "", state["table"] or "")
        except AppError as exc:
            return _err(state, exc)
        return {"response": env}

    async def select_query(state: ChatState) -> ChatState:
        caller = state["user"]
        try:
            result = await deps.run_select(caller, state["message"], state["schema"] or "")
        except AppError as exc:
            return _err(state, exc)
        except Exception as exc:
            return {"response": error_envelope(_select_error(exc))}
        return {
            "sql": result.sql,
            "response": {
                "type": "query_result",
                "data": {"columns": result.columns, "rows": result.rows, "sql": result.sql,
                         "truncated": result.truncated, "row_count": len(result.rows)},
                "message": result.answer,
            },
        }

    async def _draft(state: ChatState, previous: Any = None, error: str | None = None) -> ChatState:
        caller, intent = state["user"], Intent(state["intent"])
        schema, table, message = state["schema"], state["table"], state["message"]
        model = deps.chat_model_factory()
        assert schema is not None and table is not None
        try:
            if intent == Intent.ingest_file:
                upload_uuid, extraction = await asyncio.to_thread(
                    _load_upload_sync, caller, str(state["upload_id"]))
                if not extraction.tables:
                    raise AppError(ErrorCode.NO_TABLE_FOUND_IN_FILE)
                file_tables = [
                    {"index": i, "page": t.page, "header": t.header,
                     "rows": t.rows[: settings.llm_sample_rows], "row_count": len(t.rows)}
                    for i, t in enumerate(extraction.tables[:MAX_FILE_TABLES_FOR_LLM])
                ]
                draft = await draft_ingest(
                    message, chat_model=model, schema=schema, table=table,
                    file_tables=file_tables, raw_excerpt="" if extraction.tables else
                    extraction.raw_text_excerpt, previous=previous, error=error)
                draft = _enforce_user_intent(draft, message, schema, table)
                return {"draft": draft, "plan": ingest_to_plan(draft, extraction, upload_uuid),
                        "error": None}
            description = await asyncio.to_thread(_table_description_sync, caller, schema, table)
            plan = await draft_plan(intent, message, chat_model=model, schema=schema, table=table,
                                    table_description=description, previous=previous, error=error)
            plan = _enforce_user_intent(plan, message, schema, table)
            return {"plan": plan, "draft": plan, "error": None}
        except (LLMUnavailableError, LLMBadOutputError) as exc:
            return {"response": error_envelope(_llm_error(exc))}
        except PlanValidationError as exc:
            return {"error": {**exc.to_dict(), "repairable": exc.repairable}, "plan": None}
        except ValidationError as exc:
            msg = "; ".join(e["msg"] for e in exc.errors()[:5])
            return {"error": {"code": ErrorCode.PLAN_INVALID.value, "message": msg,
                              "repairable": True}, "plan": None}
        except AppError as exc:
            return _err(state, exc)

    async def build_plan(state: ChatState) -> ChatState:
        return {**await _draft(state), "attempts": 1}

    async def repair_plan(state: ChatState) -> ChatState:
        error = state.get("error") or {}
        out = await _draft(state, previous=state.get("draft"), error=error.get("message"))
        return {**out, "attempts": state.get("attempts", 1) + 1}

    async def validate(state: ChatState) -> ChatState:
        if state.get("response"):
            return {}
        if state.get("plan") is None:  # drafting failed with a repairable error
            return {}
        try:
            validated = await asyncio.to_thread(_validate_sync, state["plan"], state["user"])
        except PlanValidationError as exc:
            return {"error": {**exc.to_dict(), "repairable": exc.repairable}}
        except AppError as exc:
            return _err(state, exc)
        return {"validated": validated, "error": None}

    async def create_pending_operation(state: ChatState) -> ChatState:
        try:
            preview = await asyncio.to_thread(
                _create_pending_sync, state["user"], state["validated"],
                state.get("conversation_id"), state.get("upload_id"))
        except AppError as exc:
            return _err(state, exc)
        return {
            "operation_id": preview["operation_id"],
            "response": {"type": "operation_preview", "data": preview,
                         "message": preview_message(preview)},
        }

    async def respond(state: ChatState) -> ChatState:
        response = state.get("response")
        if response is None:
            error = state.get("error") or {}
            code = error.get("code") or ErrorCode.PLAN_INVALID.value
            exc = AppError(ErrorCode(code) if code in ErrorCode.__members__ else ErrorCode.PLAN_INVALID,
                           error.get("message") or "I could not build a valid plan for that.",
                           details={k: v for k, v in (error.get("details") or {}).items()
                                    if k != "repairable"})
            response = error_envelope(exc)
        if deps.record_history:
            conv = state.get("conversation_id")
            op = state.get("operation_id")
            await asyncio.to_thread(
                query_runs.record, user_id=state["user"].user_id,
                conversation_id=uuid.UUID(conv) if conv else None, message=state["message"],
                intent=state.get("intent"), envelope=response, sql=state.get("sql"),
                operation_id=uuid.UUID(op) if op else None,
            )
        return {"response": response}

    # ---- edges ----------------------------------------------------------- #
    def after_route(state: ChatState) -> str:
        if state.get("response"):
            return "respond"
        intent = Intent(state["intent"])
        if intent in METADATA_INTENTS:
            return "metadata_op"
        if intent == Intent.select_query:
            return "select_query"
        if intent in WRITE_INTENTS:
            return "build_plan"
        return "respond"

    def after_validate(state: ChatState) -> str:
        if state.get("response"):
            return "respond"
        error = state.get("error")
        if error:
            if error.get("repairable") and state.get("attempts", 1) < MAX_ATTEMPTS:
                return "repair_plan"
            return "respond"
        return "create_pending_operation"

    graph = StateGraph(ChatState)
    graph.add_node("route_intent", route_intent)
    graph.add_node("metadata_op", metadata_op)
    graph.add_node("select_query", select_query)
    graph.add_node("build_plan", build_plan)
    graph.add_node("validate_plan", validate)
    graph.add_node("repair_plan", repair_plan)
    graph.add_node("create_pending_operation", create_pending_operation)
    graph.add_node("respond", respond)

    graph.add_edge(START, "route_intent")
    graph.add_conditional_edges("route_intent", after_route, {
        "respond": "respond", "metadata_op": "metadata_op", "select_query": "select_query",
        "build_plan": "build_plan"})
    graph.add_edge("metadata_op", "respond")
    graph.add_edge("select_query", "respond")
    graph.add_edge("build_plan", "validate_plan")
    graph.add_edge("repair_plan", "validate_plan")
    graph.add_conditional_edges("validate_plan", after_validate, {
        "respond": "respond", "repair_plan": "repair_plan",
        "create_pending_operation": "create_pending_operation"})
    graph.add_edge("create_pending_operation", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


def _select_error(exc: Exception) -> AppError:
    name = type(exc).__name__
    if name == "CannotAnswer":
        return AppError(ErrorCode.QUERY_FAILED, str(exc)[:300] or "I can't answer that from the data.")
    if name == "SQLGuardError":
        return AppError(ErrorCode.QUERY_FAILED, "The generated query was blocked by the safety guard.")
    if name == "QueryFailedError":
        return AppError(ErrorCode.QUERY_FAILED, "Could not build a working query for that question.")
    if name == "ConfigurationError":
        return AppError(ErrorCode.NOT_CONFIGURED, "Natural-language queries are not configured.")
    if "Provider" in name or name in ("AgentError", "LLMUnavailableError"):
        return AppError(ErrorCode.LLM_UNAVAILABLE)
    log.exception("select_query failed", exc_info=exc)
    return AppError(ErrorCode.INTERNAL_ERROR)


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
_graph_cache: dict[int, Any] = {}


def get_graph(deps: ChatDeps | None = None) -> Any:
    key = id(deps)
    if key not in _graph_cache:
        _graph_cache[key] = build_chat_graph(deps)
    return _graph_cache[key]


def initial_state(caller: Caller, message: str, *, upload_id: str | None,
                  conversation_id: str | None) -> ChatState:
    return {"user": caller, "message": message,
            "upload_id": upload_id, "conversation_id": conversation_id, "attempts": 0}


async def run_chat(caller: Caller, message: str, *, upload_id: str | None = None,
                   conversation_id: str | None = None, deps: ChatDeps | None = None) -> dict[str, Any]:
    state = await get_graph(deps).ainvoke(
        initial_state(caller, message, upload_id=upload_id, conversation_id=conversation_id))
    return dict(state["response"])


async def stream_chat(caller: Caller, message: str, *, upload_id: str | None = None,
                      conversation_id: str | None = None,
                      deps: ChatDeps | None = None) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Yields (event, payload): status events per node, then one result or error."""
    final: dict[str, Any] | None = None
    async for update in get_graph(deps).astream(
        initial_state(caller, message, upload_id=upload_id, conversation_id=conversation_id),
        stream_mode="updates",
    ):
        for node, values in update.items():
            yield "status", {"stage": node, "message": STATUS_TEXT.get(node, node)}
            if node == "respond" and values:
                final = values.get("response")
    if final is None:
        final = error_envelope(AppError(ErrorCode.INTERNAL_ERROR))
    yield ("error" if final.get("type") == "error" else "result"), final
