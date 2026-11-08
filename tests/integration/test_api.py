"""End-to-end API tests (FastAPI TestClient + real PostgreSQL + fake LLM).

Every user has full access to their own workspace and no access to anyone
else's: operations, uploads, conversations, history, tables, rows, exports.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.security.authorization import Caller
from tests.conftest import make_user, phys
from tests.fakes import STUDENTS_DRAFT, FakeChatModel, deps_with

FIX = Path(__file__).resolve().parents[1] / "fixtures"
API = "/api/v1"
STORE = "store this as students in schema college"


@dataclass
class Session:
    headers: dict[str, str]
    caller: Caller


@pytest.fixture
def client(clean_target: Any) -> Iterator[TestClient]:
    from app.app import app

    with TestClient(app) as c:
        yield c
    app.state.chat_deps = None


def use_fake(client: TestClient, fake: FakeChatModel) -> None:
    client.app.state.chat_deps = deps_with(fake, record_history=True)  # type: ignore[attr-defined]


def login(client: TestClient) -> Session:
    user = make_user()
    r = client.post(f"{API}/auth/login", json={"email": user.email, "password": "password123"})
    assert r.status_code == 200, r.text
    assert "role" not in r.json()
    return Session({"Authorization": f"Bearer {r.json()['access_token']}"}, Caller(user.id))


def upload(client: TestClient, s: Session, name: str = "students.pdf",
           filename: str | None = None, content_type: str = "application/pdf") -> Any:
    with open(FIX / name, "rb") as fh:
        return client.post(f"{API}/files/upload", headers=s.headers,
                           files={"file": (filename or name, fh, content_type)})


def students_fake() -> FakeChatModel:
    return FakeChatModel({"IntentDecision": [{"intent": "ingest_file", "schema": "college",
                                              "table": "students"}],
                          "IngestDraft": [STUDENTS_DRAFT]})


def preview_students(client: TestClient, s: Session) -> dict[str, Any]:
    r = upload(client, s)
    assert r.status_code == 201, r.text
    use_fake(client, students_fake())
    r = client.post(f"{API}/chat?stream=false", headers=s.headers,
                    json={"message": STORE, "upload_id": r.json()["upload_id"]})
    assert r.status_code == 200, r.text
    env = r.json()
    assert env["type"] == "operation_preview", env
    return env


def stored_students(client: TestClient, s: Session) -> str:
    op_id = preview_students(client, s)["data"]["operation_id"]
    r = client.post(f"{API}/operations/{op_id}/confirm", headers=s.headers)
    assert r.status_code == 200, r.text
    return op_id


# --------------------------------------------------------------------------- #
def test_happy_path_upload_chat_confirm(client: TestClient, admin_target: Any, admin_meta: Any) -> None:
    s = login(client)
    up = upload(client, s)
    assert up.status_code == 201
    body = up.json()
    assert body["mime"] == "application/pdf" and len(body["sha256"]) == 64
    assert body["extraction"]["tables"][0]["header"][0] == "Roll No"

    env = preview_students(client, s)
    op_id = env["data"]["operation_id"]
    assert env["conversation_id"] and env["data"]["total_rows"] == 5
    assert env["data"]["target"] == {"schema": "college", "table": "students"}

    r = client.get(f"{API}/operations/{op_id}", headers=s.headers)
    assert r.json()["status"] == "pending" and r.json()["target"]["schema"] == "college"

    r = client.post(f"{API}/operations/{op_id}/confirm", headers=s.headers)
    assert r.status_code == 200, r.text
    result = r.json()
    assert result["type"] == "operation_result" and result["data"]["status"] == "executed"
    assert result["data"]["result"]["rows_written"] == 5
    assert result["message"].startswith("Done: created college.students")
    table = f'"{phys(s.caller, "college")}".students'
    rows = admin_target.execute(f"SELECT roll_no, name, gpa::text FROM {table} ORDER BY roll_no").fetchall()
    assert rows[0] == (101, "Asha Verma", "8.70") and len(rows) == 5

    events = [e for (e,) in admin_meta.execute(
        "SELECT event || ':' || status FROM audit_logs WHERE operation_id = %s ORDER BY id",
        (op_id,)).fetchall()]
    assert events == ["plan:pending", "confirm:confirmed", "execute:executed"]
    hist = client.get(f"{API}/queries", headers=s.headers).json()
    assert hist[0]["operation_id"] == op_id and hist[0]["response_type"] == "operation_preview"

    assert [t["schema"] for t in client.get(f"{API}/schemas", headers=s.headers).json()["data"]["items"]] \
        == ["college"]
    page = client.get(f"{API}/schemas/college/tables/students/rows?page=1&page_size=2&sort=gpa&dir=desc",
                      headers=s.headers).json()["data"]
    assert [r["name"] for r in page["rows"]] == ["Meera Iyer", "Asha Verma"]
    r = client.get(f"{API}/schemas/college/tables/students/csv", headers=s.headers)
    assert r.status_code == 200 and r.text.startswith("roll_no,")


def test_every_user_can_do_everything_in_own_workspace(client: TestClient) -> None:
    """No roles: a freshly registered account can upload, create, insert, update, delete, export."""
    r = client.post(f"{API}/auth/register", json={"email": "new@example.com", "password": "password123"})
    assert r.status_code == 201 and "role" not in r.json()
    r = client.post(f"{API}/auth/login", json={"email": "new@example.com", "password": "password123"})
    s = Session({"Authorization": f"Bearer {r.json()['access_token']}"}, Caller(0))
    stored_students(client, s)
    plans = [
        {"kind": "insert_rows", "schema": "college", "table": "students",
         "columns": ["roll_no", "name"], "rows": [[200, "Zoya"]]},
        {"kind": "update_rows", "schema": "college", "table": "students",
         "set": [{"column": "age", "value": 30}], "where": [{"column": "roll_no", "op": "=", "value": 200}]},
        {"kind": "delete_rows", "schema": "college", "table": "students",
         "where": [{"column": "roll_no", "op": "=", "value": 200}]},
    ]
    for plan in plans:
        r = client.post(f"{API}/operations", headers=s.headers, json={"plan": plan})
        assert r.status_code == 201, r.text
        r = client.post(f"{API}/operations/{r.json()['data']['operation_id']}/confirm", headers=s.headers)
        assert r.status_code == 200 and r.json()["data"]["status"] == "executed", r.text
    assert client.get(f"{API}/schemas/college/tables/students/csv", headers=s.headers).status_code == 200


def test_users_cannot_see_each_others_data(client: TestClient, admin_target: Any) -> None:
    alice, bob = login(client), login(client)
    op_id = stored_students(client, alice)
    upload_id = upload(client, alice).json()["upload_id"]

    # schemas / tables / rows / csv: bob's workspace is empty
    assert client.get(f"{API}/schemas", headers=bob.headers).json()["data"]["items"] == []
    for path in ("/schemas/college/tables", "/schemas/college/tables/students",
                 "/schemas/college/tables/students/rows", "/schemas/college/tables/students/csv",
                 f"/schemas/{alice.caller.prefix}college/tables/students"):
        r = client.get(f"{API}{path}", headers=bob.headers)
        assert r.status_code == 404, (path, r.text)

    # operations, uploads, history, conversations
    for method, path in (("get", f"/operations/{op_id}"), ("post", f"/operations/{op_id}/confirm"),
                         ("post", f"/operations/{op_id}/cancel"), ("get", f"/files/{upload_id}")):
        r = getattr(client, method)(f"{API}{path}", headers=bob.headers)
        assert r.status_code == 404, (path, r.text)
    assert client.get(f"{API}/queries", headers=bob.headers).json() == []
    assert client.get(f"{API}/conversations", headers=bob.headers).json() == []

    # bob can't use alice's upload or conversation in chat
    use_fake(client, students_fake())
    env = client.post(f"{API}/chat?stream=false", headers=bob.headers,
                      json={"message": STORE, "upload_id": upload_id}).json()
    assert env["type"] == "error" and env["data"]["code"] == "UPLOAD_NOT_FOUND"
    conv = client.get(f"{API}/conversations", headers=alice.headers).json()[0]["conversation_id"]
    r = client.post(f"{API}/chat?stream=false", headers=bob.headers,
                    json={"message": "show all schemas", "conversation_id": conv})
    assert r.status_code == 404

    # bob writing to "college.students" only ever reaches his own (missing) table
    r = client.post(f"{API}/operations", headers=bob.headers, json={"plan": {
        "kind": "delete_rows", "schema": "college", "table": "students", "where": [], "all_rows": True}})
    assert r.status_code == 404 and r.json()["error"]["code"] == "TABLE_NOT_FOUND"
    table = f'"{phys(alice.caller, "college")}".students'
    assert admin_target.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 5

    # same names, separate data
    stored_students(client, bob)
    for s in (alice, bob):
        items = client.get(f"{API}/schemas/college/tables/students/rows", headers=s.headers).json()
        assert items["data"]["total_rows"] == 5


def test_no_user_listing_or_role_endpoints(client: TestClient) -> None:
    s = login(client)
    assert client.get(f"{API}/auth/users", headers=s.headers).status_code in (404, 405)
    assert client.patch(f"{API}/auth/users/1/role", headers=s.headers,
                        json={"role": "admin"}).status_code in (404, 405)
    me = client.get(f"{API}/auth/me", headers=s.headers).json()
    assert set(me) == {"id", "email", "is_active", "created_at"}


def test_legacy_ai_routes_not_mounted(client: TestClient) -> None:
    """The original /ai routes query one shared schema, so they are off by default."""
    s = login(client)
    assert client.post(f"{API}/ai/chat", headers=s.headers, json={"message": "hi"}).status_code == 404


def test_double_confirm_is_idempotent(client: TestClient, admin_target: Any, admin_meta: Any) -> None:
    s = login(client)
    op_id = preview_students(client, s)["data"]["operation_id"]
    first = client.post(f"{API}/operations/{op_id}/confirm", headers=s.headers)
    second = client.post(f"{API}/operations/{op_id}/confirm", headers=s.headers)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    table = f'"{phys(s.caller, "college")}".students'
    assert admin_target.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 5
    executions = admin_meta.execute("SELECT count(*) FROM audit_logs WHERE operation_id=%s "
                                    "AND event='execute'", (op_id,)).fetchone()[0]
    assert executions == 1


def test_expired_operation_is_rejected(client: TestClient, admin_meta: Any, admin_target: Any) -> None:
    s = login(client)
    op_id = preview_students(client, s)["data"]["operation_id"]
    admin_meta.execute("UPDATE operations SET expires_at = now() - interval '1 minute' "
                       "WHERE id = %s", (op_id,))
    r = client.post(f"{API}/operations/{op_id}/confirm", headers=s.headers)
    assert r.status_code == 410 and r.json()["error"]["code"] == "OPERATION_EXPIRED"
    assert client.get(f"{API}/operations/{op_id}", headers=s.headers).json()["status"] == "expired"
    assert admin_target.execute("SELECT count(*) FROM pg_namespace WHERE nspname = %s",
                                (phys(s.caller, "college"),)).fetchone()[0] == 0


def test_cancel_then_confirm(client: TestClient) -> None:
    s = login(client)
    op_id = preview_students(client, s)["data"]["operation_id"]
    assert client.post(f"{API}/operations/{op_id}/cancel", headers=s.headers).status_code == 200
    r = client.post(f"{API}/operations/{op_id}/confirm", headers=s.headers)
    assert r.status_code == 409 and r.json()["error"]["code"] == "OPERATION_NOT_PENDING"


def test_sse_stream(client: TestClient) -> None:
    s = login(client)
    use_fake(client, FakeChatModel())
    with client.stream("POST", f"{API}/chat", headers={**s.headers, "Accept": "text/event-stream"},
                       json={"message": "show me all schemas"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    events = [line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("event:")]
    datas = [json.loads(line.split(":", 1)[1]) for line in text.splitlines() if line.startswith("data:")]
    assert events[0] == "status" and events[-1] == "result"
    assert datas[-1]["type"] == "schema_list" and datas[-1]["conversation_id"]


@pytest.mark.parametrize(("name", "code"), [("scanned.pdf", "SCANNED_PDF_UNSUPPORTED"),
                                            ("no_tables.pdf", "NO_TABLE_FOUND_IN_FILE")])
def test_unusable_pdfs(client: TestClient, name: str, code: str) -> None:
    r = upload(client, login(client), name)
    assert r.status_code == 422 and r.json()["error"]["code"] == code


def test_upload_type_checked_by_magic_bytes(client: TestClient) -> None:
    s = login(client)
    r = upload(client, s, "students.pdf", filename="students.csv", content_type="text/csv")
    assert r.status_code == 201 and r.json()["mime"] == "application/pdf"
    r = client.post(f"{API}/files/upload", headers=s.headers,
                    files={"file": ("evil.pdf", b"MZ\x90\x00\x03binary", "application/pdf")})
    assert r.status_code == 415 and r.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"


def test_system_and_shared_schemas_unreachable(client: TestClient) -> None:
    s = login(client)
    for schema in ("pg_catalog", "information_schema"):
        r = client.get(f"{API}/schemas/{schema}/tables", headers=s.headers)
        assert r.status_code in (403, 422), (schema, r.text)
    for schema in ("public", "private"):  # these mean the user's own (missing) schemas
        r = client.get(f"{API}/schemas/{schema}/tables", headers=s.headers)
        assert r.status_code == 404, (schema, r.text)
    use_fake(client, FakeChatModel())
    env = client.post(f"{API}/chat?stream=false", headers=s.headers,
                      json={"message": "show tables in schema pg_catalog"}).json()
    assert env["type"] == "error" and env["data"]["code"] in ("FORBIDDEN_SCHEMA", "INVALID_IDENTIFIER")


def test_auth_required_and_errors_are_clean(client: TestClient) -> None:
    r = client.get(f"{API}/schemas")
    assert r.status_code == 401 and r.json()["error"]["code"] == "NOT_AUTHENTICATED"
    r = client.get(f"{API}/schemas", headers={"Authorization": "Bearer garbage"})
    assert r.status_code == 401
    r = client.post(f"{API}/auth/login", json={"email": "nobody@example.com", "password": "x"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "INVALID_CREDENTIALS"


def test_create_table_form_flow(client: TestClient, admin_target: Any) -> None:
    """POST /operations (the frontend "Create table" form) -> preview -> confirm."""
    s = login(client)
    plan = {"kind": "create_table", "schema": "main", "table": "courses",
            "columns": [{"name": "id", "pg_type": "integer", "primary_key": True},
                        {"name": "title", "pg_type": "varchar(100)", "nullable": False}]}
    r = client.post(f"{API}/operations", headers=s.headers, json={"plan": plan})
    assert r.status_code == 201, r.text
    env = r.json()
    assert env["data"]["ddl"][:2] == ["CREATE SCHEMA main", env["data"]["ddl"][1]]
    assert "CREATE TABLE main.courses" in env["data"]["ddl"][1]
    r = client.post(f"{API}/operations/{env['data']['operation_id']}/confirm", headers=s.headers)
    assert r.status_code == 200 and r.json()["data"]["status"] == "executed"
    assert admin_target.execute("SELECT to_regclass(%s)",
                                (f'"{phys(s.caller, "main")}".courses',)).fetchone()[0] is not None
    bad = {**plan, "schema": "pg_catalog"}
    r = client.post(f"{API}/operations", headers=s.headers, json={"plan": bad})
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_audit_log_is_append_only(admin_meta: Any, db_required: None) -> None:
    import psycopg

    admin_meta.execute("INSERT INTO audit_logs (event, status) VALUES ('test', 'ok')")
    for sql in ("UPDATE audit_logs SET status = 'x'", "DELETE FROM audit_logs", "TRUNCATE audit_logs"):
        with pytest.raises(psycopg.errors.RaiseException):
            admin_meta.execute(sql)
