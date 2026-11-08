-- Runs once, when the docker compose Postgres volume is first created
-- (mounted into /docker-entrypoint-initdb.d by compose.yaml).
--
-- Creates the two databases, which must stay separate:
--   meta_db   app internals (users, uploads, operations, audit_logs, query_history)
--   target_db the user's data that chat operates on
--
-- Then:
--   alembic upgrade head                                        (META_DATABASE_URL -> meta_db)
--   python scripts/setup_roles.py --admin-url <superuser url to target_db>   (vaak_ro / vaak_rw)

SELECT 'CREATE DATABASE meta_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'meta_db')\gexec

SELECT 'CREATE DATABASE target_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'target_db')\gexec
