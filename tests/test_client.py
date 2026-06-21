"""Tests for FMPClient 429 retry/backoff. Fully offline — httpx is monkeypatched and
no real network or sleep happens (tenacity's backoff sleep is stubbed)."""
from __future__ import annotations

import httpx
import pytest

from startx.fmp.client import FMPClient, FMPError, FMPRateLimited
from startx.settings import Settings


def _client(monkeypatch):
    """Build an FMPClient with a fake key and instant (no-sleep) retry backoff."""
    # Neutralise tenacity's exponential sleep so retries don't actually wait.
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda *_: None)
    return FMPClient(settings=Settings(fmp_api_key="test-key"))


def _resp(status: int, json_body=None) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        json=json_body if json_body is not None else [],
        request=httpx.Request("GET", "https://example.test/x"),
    )


def test_429_retries_then_succeeds(monkeypatch):
    """A transient 429 is retried and a subsequent 200 is returned (no FMPError)."""
    client = _client(monkeypatch)
    calls = []

    def fake_get(url, params=None):
        calls.append(url)
        # 429 on the first two attempts, then a clean 200.
        if len(calls) < 3:
            return _resp(429)
        return _resp(200, [{"ok": True}])

    monkeypatch.setattr(client._client, "get", fake_get)
    out = client.get("historical-price-eod/full", symbol="SPY")
    assert out == [{"ok": True}]
    assert len(calls) == 3  # 2 retried 429s + 1 success


def test_429_retries_are_bounded_then_raises(monkeypatch):
    """Persistent 429s exhaust the bounded retries and surface as FMPError (not a hang)."""
    client = _client(monkeypatch)
    calls = []

    def always_429(url, params=None):
        calls.append(url)
        return _resp(429)

    monkeypatch.setattr(client._client, "get", always_429)
    with pytest.raises(FMPError) as exc:
        client.get("historical-price-eod/full", symbol="SPY")
    assert "429" in str(exc.value)
    # stop_after_attempt(4) -> exactly 4 attempts, then stop. Bounded, no infinite loop.
    assert len(calls) == 4


def test_429_raised_inside_send_is_retryable_type(monkeypatch):
    """_send raises the retryable FMPRateLimited (not FMPError) on a raw 429 response."""
    client = _client(monkeypatch)

    monkeypatch.setattr(client._client, "get", lambda url, params=None: _resp(429))
    with pytest.raises(FMPRateLimited):
        client._send("x", {})


def test_transport_error_still_retries(monkeypatch):
    """Existing transport-fault retry path is unchanged: a TransportError retries then 200."""
    client = _client(monkeypatch)
    calls = []

    def flaky(url, params=None):
        calls.append(url)
        if len(calls) < 2:
            raise httpx.ConnectError("boom")
        return _resp(200, [{"ok": True}])

    monkeypatch.setattr(client._client, "get", flaky)
    assert client.get("x") == [{"ok": True}]
    assert len(calls) == 2


def test_403_does_not_retry(monkeypatch):
    """Non-retryable statuses (e.g. 403 plan limit) raise immediately, no retry loop."""
    client = _client(monkeypatch)
    calls = []

    def forbidden(url, params=None):
        calls.append(url)
        return _resp(403)

    monkeypatch.setattr(client._client, "get", forbidden)
    with pytest.raises(FMPError) as exc:
        client.get("grades", symbol="SPY")
    assert "403" in str(exc.value)
    assert len(calls) == 1  # raised on the first attempt, not retried
