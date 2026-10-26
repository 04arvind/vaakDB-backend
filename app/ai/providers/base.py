"""Provider interface and the shared HTTP plumbing.

Providers talk to vendor REST APIs with httpx directly (no vendor SDKs), so
timeouts, retries and error mapping behave the same for every provider.

Error mapping:
    401/403      -> ProviderAuthError        (not retried; fallback may try another provider)
    429          -> ProviderRateLimitError   (retried, honours Retry-After)
    408/5xx/net  -> ProviderUnavailableError (retried)
    timeout      -> ProviderTimeoutError     (retried)
    other 4xx    -> ProviderError            (not retried: the request itself is wrong)
"""
from __future__ import annotations

import abc
import logging
import time
from typing import Any

import httpx

from app.ai.exceptions import AIError
from app.ai.schemas.llm_io import LLMResponse, Message
from app.ai.utils.retry import retry_async

log = logging.getLogger(__name__)


class ProviderError(AIError):
    """An LLM provider failed. Messages may contain vendor detail: log, don't show."""

    retryable: bool = False

    def __init__(self, message: str, *, provider: str = "", status_code: int | None = None):
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"[{provider}] {message}" if provider else message)


class ProviderNotConfiguredError(ProviderError):
    """API key missing or provider unknown."""


class ProviderAuthError(ProviderError):
    """The API key was rejected."""


class ProviderRateLimitError(ProviderError):
    retryable = True

    def __init__(self, message: str, *, retry_after: float | None = None, **kw: Any):
        self.retry_after = retry_after
        super().__init__(message, **kw)


class ProviderUnavailableError(ProviderError):
    retryable = True


class ProviderTimeoutError(ProviderUnavailableError):
    pass


class ContentBlockedError(ProviderError):
    """The provider refused the prompt or the answer (safety filters)."""


_RETRYABLE = (ProviderRateLimitError, ProviderUnavailableError)


class LLMProvider(abc.ABC):
    name: str = "base"
    supports_attachments: bool = False
    supports_embeddings: bool = False

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        default_model: str,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.max_retries = max_retries
        self._timeout = httpx.Timeout(timeout_s, connect=10.0)
        self._transport = transport  # injectable for tests (httpx.MockTransport)
        self._client: httpx.AsyncClient | None = None

    # ---- public API -------------------------------------------------------- #
    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    @abc.abstractmethod
    async def generate(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResponse: ...

    async def embed(
        self, texts: list[str], *, task_type: str | None = None
    ) -> list[list[float]]:
        """task_type: "RETRIEVAL_DOCUMENT" for stored chunks, "RETRIEVAL_QUERY" for queries."""
        raise ProviderError("Embeddings are not supported.", provider=self.name)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ---- helpers for subclasses ------------------------------------------- #
    def _headers(self) -> dict[str, str]:
        return {}

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout,
                transport=self._transport,
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
            )
        return self._client

    def _require_key(self) -> None:
        if not self._api_key:
            raise ProviderNotConfiguredError("API key is not configured.", provider=self.name)

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with retries and uniform error mapping. Returns the decoded JSON body."""
        self._require_key()

        async def once() -> dict[str, Any]:
            try:
                resp = await self._http().post(path, json=payload, headers=self._headers())
            except httpx.TimeoutException as exc:
                raise ProviderTimeoutError("Request timed out.", provider=self.name) from exc
            except httpx.TransportError as exc:
                raise ProviderUnavailableError(
                    f"Network error: {type(exc).__name__}", provider=self.name
                ) from exc
            self._raise_for_status(resp)
            try:
                return resp.json()
            except ValueError as exc:
                raise ProviderUnavailableError(
                    "Provider returned invalid JSON.", provider=self.name
                ) from exc

        return await retry_async(once, attempts=self.max_retries + 1, retry_on=_RETRYABLE)

    def _raise_for_status(self, resp: httpx.Response) -> None:
        code = resp.status_code
        if code < 400:
            return
        detail = _error_detail(resp)
        kw = {"provider": self.name, "status_code": code}
        if code in (401, 403):
            raise ProviderAuthError(f"Authentication failed ({code}).", **kw)
        if code == 429:
            raise ProviderRateLimitError(
                f"Rate limited: {detail}", retry_after=_retry_after(resp), **kw
            )
        if code == 408 or code >= 500:
            raise ProviderUnavailableError(f"Upstream error {code}: {detail}", **kw)
        raise ProviderError(f"Request rejected ({code}): {detail}", **kw)

    @staticmethod
    def _now_ms() -> float:
        return time.perf_counter() * 1000


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None  # HTTP-date form; fall back to backoff


def _error_detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        err = body.get("error", body) if isinstance(body, dict) else body
        msg = err.get("message") if isinstance(err, dict) else err
        return str(msg)[:300]
    except ValueError:
        return resp.text[:300]
