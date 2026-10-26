"""Async retry with exponential backoff and full jitter.

    result = await retry_async(
        lambda: client.post(...),
        attempts=3,
        retry_on=(ProviderTimeoutError, ProviderUnavailableError),
    )

If the raised exception has a `retry_after` attribute (seconds, e.g. from a
429 Retry-After header) that delay is honoured, capped at max_delay.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")
log = logging.getLogger(__name__)


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Call `fn` up to `attempts` times. Re-raises the last error."""
    attempts = max(1, attempts)
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except retry_on as exc:
            if attempt == attempts:
                raise
            hinted = getattr(exc, "retry_after", None)
            if isinstance(hinted, (int, float)) and hinted > 0:
                delay = min(float(hinted), max_delay)
            else:
                delay = random.uniform(0, min(max_delay, base_delay * 2 ** (attempt - 1)))
            log.warning(
                "Attempt %d/%d failed (%s: %s); retrying in %.2fs",
                attempt, attempts, type(exc).__name__, exc, delay,
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
