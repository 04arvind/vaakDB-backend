-- READ-ONLY role for target_db (vaak_ro). Used for metadata, previews, paginated
-- rows, CSV export and natural-language SELECT.
--
-- Run as a superuser while connected to target_db (or: python scripts/setup_roles.py).
-- Every session of this role is read-only by default and has a short statement_timeout.
-- It can read the workspace schemas created by vaak_rw (default privileges);
-- the backend decides which user's schema a request may read.

DO $do$
DECLARE
    ro_role  text := '{{RO_ROLE}}';
    ro_pass  text := '{{RO_PASSWORD}}';
    rw_role  text := '{{RW_ROLE}}';
BEGIN
    IF ro_role !~ '^[a-z_][a-z0-9_]{0,62}$' THEN
        RAISE EXCEPTION 'invalid role name %', ro_role;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = ro_role) THEN
        EXECUTE format('CREATE ROLE %I LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE '
                       'NOREPLICATION NOBYPASSRLS', ro_role, ro_pass);
    ELSE
        EXECUTE format('ALTER ROLE %I LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE '
                       'NOREPLICATION NOBYPASSRLS', ro_role, ro_pass);
    END IF;
    EXECUTE format('ALTER ROLE %I SET default_transaction_read_only = on', ro_role);
    EXECUTE format('ALTER ROLE %I SET statement_timeout = %L', ro_role, '10s');

    EXECUTE format('REVOKE ALL ON DATABASE %I FROM %I', current_database(), ro_role);
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), ro_role);

    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = rw_role) THEN
        EXECUTE format('ALTER DEFAULT PRIVILEGES FOR ROLE %I GRANT USAGE ON SCHEMAS TO %I',
                       rw_role, ro_role);
        EXECUTE format('ALTER DEFAULT PRIVILEGES FOR ROLE %I GRANT SELECT ON TABLES TO %I',
                       rw_role, ro_role);
    END IF;
END
$do$;
