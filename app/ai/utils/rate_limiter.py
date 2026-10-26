"""In-process token-bucket rate limiter, keyed by user (or IP).

    limiter = RateLimiter(per_minute=30, burst=10)
    limiter.check(user_id)        # raises RateLimitExceeded when empty

Limits are per worker process. With several uvicorn/gunicorn workers the real
limit is per_minute * workers; put a shared limiter (Redis, API gateway) in
front if that matters.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from app.ai.exceptions import RateLimitExceeded


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    def __init__(self, per_minute: int, burst: int, *, max_keys: int = 10_000):
        self.rate = per_minute / 60.0  # tokens per second
        self.burst = max(1, burst)
        self.max_keys = max_keys
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.rate > 0

    def check(self, key: str, cost: float = 1.0) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.pop(key, None) or _Bucket(float(self.burst), now)
            bucket.tokens = min(self.burst, bucket.tokens + (now - bucket.updated) * self.rate)
            bucket.updated = now
            self._buckets[key] = bucket  # most recently used goes last
            while len(self._buckets) > self.max_keys:
                self._buckets.popitem(last=False)

            if bucket.tokens < cost:
                raise RateLimitExceeded((cost - bucket.tokens) / self.rate)
            bucket.tokens -= cost

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()
