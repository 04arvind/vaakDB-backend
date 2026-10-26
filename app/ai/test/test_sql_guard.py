"""The SQL guard is the most security-critical code in the module: test it hard."""
from __future__ import annotations

import pytest

from app.ai.db.sql_guard import SQLGuardError, validate_sql

ALLOWED = {"orders", "customers", "products", "users"}
HIDDEN = {"ai_messages", "ai_memories", "alembic_version"}


def ok(sql: str, **kw) -> str:
    return validate_sql(sql, ALLOWED, blocked_tables=HIDDEN, **kw)


def blocked(sql: str, **kw) -> str:
    with pytest.raises(SQLGuardError) as info:
        ok(sql, **kw)
    return str(info.value)


# ---- allowed --------------------------------------------------------------- #
@pytest.mark.parametrize("sql", [
    "SELECT id, total FROM orders",
    "SELECT o.id, c.name FROM orders o JOIN customers c ON c.id = o.customer_id",
    "SELECT count(*) FROM orders",
    "SELECT * FROM orders",
    "SELECT o.* FROM orders o",
    "WITH recent AS (SELECT id FROM orders) SELECT count(*) FROM recent",
    "SELECT id FROM orders UNION SELECT id FROM customers",
    "SELECT status, sum(total) FROM orders GROUP BY status ORDER BY 2 DESC",
    "SELECT id FROM public.orders",
    "SELECT date_trunc('month', created_at) AS m, count(*) FROM orders GROUP BY 1",
    "SELECT name FROM products WHERE name ILIKE '%chair%'",
])
def test_allows_safe_selects(sql):
    assert ok(sql)


def test_adds_limit_when_missing():
    assert "LIMIT 100" in ok("SELECT id FROM orders")


def test_keeps_small_limit_and_caps_large_one():
    assert "LIMIT 5" in ok("SELECT id FROM orders LIMIT 5")
    out = ok("SELECT id FROM orders LIMIT 100000", max_rows=50)
    assert "LIMIT 50" in out and "100000" not in out


def test_replaces_fetch_first():
    out = ok("SELECT id FROM orders FETCH FIRST 1000 ROWS ONLY", max_rows=10)
    assert "LIMIT 10" in out and "1000" not in out


def test_strips_trailing_semicolon_single_statement():
    assert ok("SELECT id FROM orders;")


# ---- blocked --------------------------------------------------------------- #
@pytest.mark.parametrize("sql", [
    "DELETE FROM orders",
    "UPDATE orders SET total = 0",
    "INSERT INTO orders (id) VALUES (1)",
    "DROP TABLE orders",
    "TRUNCATE orders",
    "ALTER TABLE orders ADD COLUMN x int",
    "CREATE TABLE x (id int)",
    "GRANT ALL ON orders TO public",
    "COPY orders TO '/tmp/x'",
    "SET statement_timeout = 0",
])
def test_blocks_non_select(sql):
    blocked(sql)


def test_blocks_multiple_statements():
    assert "one SQL statement" in blocked("SELECT 1; DROP TABLE orders")


def test_blocks_data_modifying_cte():
    blocked("WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d")


def test_blocks_select_into_and_locking():
    blocked("SELECT * INTO new_table FROM orders")
    blocked("SELECT * FROM orders FOR UPDATE")


@pytest.mark.parametrize("sql", [
    "SELECT pg_sleep(10)",
    "SELECT pg_read_file('/etc/passwd')",
    "SELECT set_config('statement_timeout', '0', false)",
    "SELECT current_setting('data_directory')",
    "SELECT lo_import('/etc/passwd')",
    "SELECT * FROM dblink('host=x', 'select 1') AS t(a int)",
    "SELECT nextval('orders_id_seq')",
    "SELECT setval('orders_id_seq', 1)",
    "SELECT query_to_xml('select * from users', true, true, '')",
    "SELECT version()",
])
def test_blocks_dangerous_functions(sql):
    blocked(sql)


def test_blocks_schema_qualified_functions():
    blocked("SELECT pg_catalog.pg_sleep(1)")
    blocked("SELECT public.some_definer_fn(1) FROM orders")


@pytest.mark.parametrize("sql", [
    "SELECT * FROM pg_catalog.pg_user",
    "SELECT * FROM information_schema.tables",
    "SELECT * FROM pg_shadow",
    "SELECT * FROM ai_messages",
    "SELECT * FROM secret_table",
    "SELECT * FROM otherdb.public.orders",
    "SELECT * FROM generate_series(1, 10)",
])
def test_blocks_tables_outside_allow_list(sql):
    blocked(sql)


def test_blocks_cte_named_like_hidden_table():
    assert "CTE name" in blocked(
        "WITH ai_messages AS (SELECT id FROM orders) SELECT * FROM ai_messages"
    )


@pytest.mark.parametrize("sql", [
    "SELECT password_hash FROM users",
    "SELECT u.api_key FROM users u",
    "SELECT id FROM users WHERE reset_token = 'x'",
    'SELECT "PASSWORD" FROM users',
])
def test_blocks_sensitive_columns(sql):
    blocked(sql)


@pytest.mark.parametrize("sql", [
    "SELECT to_jsonb(u) FROM users u",
    "SELECT row_to_json(u.*) FROM users u",
    "SELECT u::text FROM users u",
    "SELECT array_agg(u) FROM users u",
    "SELECT ROW(u.*) FROM users u",
    "SELECT (SELECT u FROM users u LIMIT 1)",
    "SELECT s FROM (SELECT * FROM users) s",
    "SELECT concat_ws(',', u.*) FROM users u",
])
def test_blocks_whole_row_references(sql):
    blocked(sql)


def test_blocks_garbage_and_oversized_input():
    blocked("")
    blocked("   ")
    blocked("SELEC id FROM orders WHERE")
    blocked("SELECT 1" + " " * 9000)
    blocked("SELECT id FROM orders\x00")
