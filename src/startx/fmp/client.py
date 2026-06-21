"""Rate-limited, retrying HTTP client for the FMP stable API.

Synchronous on purpose: for a small seed universe correctness and easy verification beat
throughput, and 2,800 req/min is far above what the engine needs. The public surface
(``client.get(path, **params)``) is small, so swapping in an async client later is local.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..settings import Settings, get_settings


class FMPError(RuntimeError):
    """Raised for API-level errors (bad key, plan limits, malformed responses)."""


class FMPRateLimited(Exception):
    """Internal: a 429 from FMP. Raised inside ``_send`` so the retry decorator can
    back off and retry, instead of surfacing immediately. Bounded by ``stop_after_attempt``
    — once retries are exhausted it is converted to a user-facing ``FMPError``.
    """


class _RateLimiter:
    """Sliding-window limiter: at most ``max_per_min`` calls in any 60s window."""

    def __init__(self, max_per_min: int) -> None:
        self.max = max(1, int(max_per_min))
        self._calls: deque[float] = deque()

    def acquire(self) -> None:
        now = time.monotonic()
        while self._calls and now - self._calls[0] > 60.0:
            self._calls.popleft()
        if len(self._calls) >= self.max:
            sleep_for = 60.0 - (now - self._calls[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._calls.append(time.monotonic())


# 429 is retryable (transient rate-limit) via FMPRateLimited; the others are transport
# faults. All retries are bounded by stop_after_attempt below — no infinite loop.
_RETRYABLE = (httpx.TransportError, httpx.RemoteProtocolError, FMPRateLimited)


class FMPClient:
    def __init__(self, settings: Settings | None = None, timeout: float = 30.0) -> None:
        self.settings = settings or get_settings()
        self._key = self.settings.require_key()
        self._client = httpx.Client(
            base_url=self.settings.fmp_base_url.rstrip("/"),
            timeout=timeout,
            headers={"Accept": "application/json"},
        )
        self._rl = _RateLimiter(self.settings.rate_limit_per_min)

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "FMPClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- core --------------------------------------------------------------
    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=2, max=16),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _send(self, path: str, params: dict[str, Any]) -> httpx.Response:
        self._rl.acquire()
        resp = self._client.get("/" + path.lstrip("/"), params=params)
        # Raise inside the retry boundary so a transient 429 backs off and retries
        # (bounded by stop_after_attempt) instead of immediately raising FMPError and
        # being swallowed into an empty frame by the catalyst `_safe` wrapper.
        if resp.status_code == 429:
            raise FMPRateLimited(f"429 Rate limited for {path}")
        return resp

    def get(self, path: str, **params: Any) -> Any:
        """GET ``path`` with query ``params`` (apikey injected). Returns parsed JSON."""
        params = {k: v for k, v in params.items() if v is not None}
        params["apikey"] = self._key
        try:
            resp = self._send(path, params)
        except FMPRateLimited as exc:
            # Bounded retries (see _send / stop_after_attempt) are exhausted: surface as
            # a user-facing FMPError, preserving the original 429 behaviour.
            raise FMPError(
                "429 Rate limited — retries exhausted; lower RATE_LIMIT_PER_MIN."
            ) from exc

        if resp.status_code == 401:
            raise FMPError("401 Unauthorized — check FMP_API_KEY.")
        if resp.status_code == 403:
            raise FMPError(f"403 Forbidden — endpoint not in your plan: {path}")
        if resp.status_code >= 400:
            raise FMPError(f"HTTP {resp.status_code} for {path}: {resp.text[:200]}")

        try:
            data = resp.json()
        except ValueError as exc:  # pragma: no cover - defensive
            raise FMPError(f"Non-JSON response for {path}: {resp.text[:200]}") from exc

        if isinstance(data, dict) and ("Error Message" in data or "error" in data):
            raise FMPError(f"FMP error for {path}: {data}")
        return data
