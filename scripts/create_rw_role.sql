-- Restricted WRITE role for target_db (vaak_rw).
--
-- Run as a superuser (or the database owner) while connected to target_db.
-- Easiest: python scripts/setup_roles.py  (fills the placeholders for you).
--
-- Every user has a private workspace made of schemas named u<user_id>_<name>.
-- The backend creates those schemas (as vaak_rw) the first time a user creates a
-- table in them, and only ever addresses the signed-in user's own schemas.
--
-- What vaak_rw can do:
--   * CONNECT to this database and CREATE schemas in it (workspace schemas)
--   * create tables in the schemas it owns, change their rows, drop them
-- What it cannot do: superuser, create databases/roles, replicate, bypass RLS,
-- write to public, or touch schemas it doesn't own (it is granted nothing there).

DO $do$
DECLARE
    rw_role  text := '{{RW_ROLE}}';
    rw_pass  text := '{{RW_PASSWORD}}';
    ro_role  text := '{{RO_ROLE}}';
BEGIN
    IF rw_role !~ '^[a-z_][a-z0-9_]{0,62}$' THEN
        RAISE EXCEPTION 'invalid role name %', rw_role;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = rw_role) THEN
        EXECUTE format('CREATE ROLE %I LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE '
                       'NOREPLICATION NOBYPASSRLS', rw_role, rw_pass);
    ELSE
        EXECUTE format('ALTER ROLE %I LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE '
                       'NOREPLICATION NOBYPASSRLS', rw_role, rw_pass);
    END IF;
    EXECUTE format('ALTER ROLE %I SET statement_timeout = %L', rw_role, '60s');
    EXECUTE format('ALTER ROLE %I SET lock_timeout = %L', rw_role, '10s');

    EXECUTE format('REVOKE ALL ON DATABASE %I FROM %I', current_database(), rw_role);
    -- CREATE on the database = may create schemas (the workspace schemas)
    EXECUTE format('GRANT CONNECT, CREATE ON DATABASE %I TO %I', current_database(), rw_role);
    -- Postgres 15+ already does this; make sure nobody can create objects in public.
    REVOKE CREATE ON SCHEMA public FROM PUBLIC;

    -- The read role can read every workspace schema/table vaak_rw creates.
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = ro_role) THEN
        EXECUTE format('ALTER DEFAULT PRIVILEGES FOR ROLE %I GRANT USAGE ON SCHEMAS TO %I',
                       rw_role, ro_role);
        EXECUTE format('ALTER DEFAULT PRIVILEGES FOR ROLE %I GRANT SELECT ON TABLES TO %I',
                       rw_role, ro_role);
    END IF;
END
$do$;
