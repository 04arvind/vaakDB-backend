"""Database URL helpers.

normalize_pg_url():
- switches postgres:// / postgresql:// (and psycopg2) to the installed psycopg 3 driver
- percent-encodes the user and password, so a password such as "pa@ss" works
  (SQLAlchemy would otherwise read "ss@localhost" as the host)

mask_url() hides the password for logs and API responses.
"""
from __future__ import annotations

from urllib.parse import quote, unquote

_DRIVER = "postgresql+psycopg"
_PLAIN = {"postgres", "postgresql", "postgresql+psycopg2", "postgresql+psycopg"}


def _split(url: str) -> tuple[str, str, str, str]:
    url = url.strip().strip('"').strip("'")
    scheme, sep, rest = url.partition("://")
    if not sep:
        raise ValueError("Database URL must look like postgresql://user:password@host/db")
    # Credentials end at the LAST '@' before the query string, so an unescaped
    # '@' (or '/') inside the password is kept in the password.
    query_at = rest.find("?")
    head = rest if query_at < 0 else rest[:query_at]
    at = head.rfind("@")
    creds, host_part = (rest[:at], rest[at + 1:]) if at >= 0 else ("", rest)
    host, slash, tail = host_part.partition("/")
    return scheme, creds, host, slash + tail


def normalize_pg_url(url: str) -> str:
    scheme, creds, host, path = _split(url)
    if scheme.lower() in _PLAIN:
        scheme = _DRIVER
    if creds:
        user, colon, password = creds.partition(":")
        creds = quote(unquote(user), safe="")
        if colon:
            creds += ":" + quote(unquote(password), safe="")
        return f"{scheme}://{creds}@{host}{path}"
    return f"{scheme}://{host}{path}"


def mask_url(url: str | None) -> str | None:
    if not url:
        return url
    try:
        scheme, creds, host, path = _split(url)
    except ValueError:
        return "<invalid url>"
    user = creds.partition(":")[0]
    return f"{scheme}://{unquote(user) + ':***@' if creds else ''}{host}{path.split('?')[0]}"
