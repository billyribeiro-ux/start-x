"""Local Parquet cache so we never re-pull the same history from FMP.

Coverage & TTL (added to fix a silent truncated-history bug)
------------------------------------------------------------
The price cache key (e.g. ``prices/SPY``) carries the symbol but historically not
the date range. That meant a cache first populated with a *short* window (say
2019->today) would be served verbatim even when a caller later asked for a *wider*
window (2010->today), silently returning truncated history and never re-fetching.

``get_or_fetch`` now accepts an optional requested coverage range
(``coverage_start`` / ``coverage_end``) and an optional ``max_age`` TTL. When a
range is requested it is compared against the *covered* range recorded in a sidecar
metadata file written at save time; if the cache does not fully cover the request
(or is older than ``max_age``) the cache is bypassed and ``fetch`` is re-run. All of
this is opt-in: callers that pass only ``key``/``fetch``/``refresh`` get the original
behaviour unchanged.

Residual risk: caches written before this change (or by ``save`` calls that don't go
through ``get_or_fetch`` with a coverage range) have no sidecar metadata. For those,
coverage is *inferred* from a ``date``/``ts`` column when one exists; if no such column
exists and a coverage range is requested, the legacy frame is served with a warning
rather than silently truncating.
"""
from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path
from typing import Callable

import pandas as pd

_SAFE = re.compile(r"[^A-Za-z0-9_-]")
# Columns we treat as the time axis when inferring coverage from a legacy frame.
_DATE_COLS = ("date", "ts", "datetime", "timestamp")


def _frame_span(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Best-effort ``(min, max)`` of the time axis of ``df``; ``None`` if undeterminable."""
    if df is None or df.empty:
        return None
    for col in _DATE_COLS:
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce").dropna()
            if not ts.empty:
                return ts.min(), ts.max()
    return None


class ParquetCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        parts = [_SAFE.sub("_", p) for p in key.split("/") if p]
        path = self.dir.joinpath(*parts).with_suffix(".parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _meta_path(self, key: str) -> Path:
        return self._path(key).with_suffix(".meta.json")

    def load(self, key: str) -> pd.DataFrame | None:
        path = self._path(key)
        return pd.read_parquet(path) if path.exists() else None

    def _read_meta(self, key: str) -> dict | None:
        mpath = self._meta_path(key)
        if not mpath.exists():
            return None
        try:
            return json.loads(mpath.read_text())
        except (OSError, ValueError):
            return None

    def save(
        self,
        key: str,
        df: pd.DataFrame,
        *,
        coverage_start: str | None = None,
        coverage_end: str | None = None,
    ) -> None:
        if df is None:
            return
        path = self._path(key)
        df.to_parquet(path, index=False)
        # Record coverage metadata so a later wider-range request can detect that this
        # cache is too narrow.
        #
        # start: record the *requested* start as the covered floor. A provider's first
        #   available bar is often a day or two after the requested start (e.g. asked for
        #   2010-01-01, first bar 2010-01-04); recording the request avoids a pointless
        #   re-fetch on every subsequent same-floor request. Fall back to the frame axis
        #   when no request was given.
        # end: record the *actual* last bar, clamped to the requested end. The tail is
        #   where new data appears, so the metadata must reflect what the cache truly
        #   holds — claiming coverage through a requested end the provider hasn't filled
        #   yet would mask a missing fresh bar forever. We never record an end *beyond*
        #   the requested window either (no false claim past what was asked for).
        span = _frame_span(df)
        if coverage_start is None and span is not None:
            coverage_start = span[0].isoformat()
        actual_end = span[1] if span is not None else None
        if actual_end is not None:
            if coverage_end is not None:
                coverage_end = min(pd.Timestamp(coverage_end), actual_end).isoformat()
            else:
                coverage_end = actual_end.isoformat()
        # else: empty/axis-less frame -> keep the requested coverage_end (or None).
        meta = {
            "saved_at": time.time(),
            "rows": int(len(df)),
            "coverage_start": coverage_start,
            "coverage_end": coverage_end,
        }
        try:
            self._meta_path(key).write_text(json.dumps(meta))
        except OSError:
            pass  # metadata is an optimisation; never fail a save over it

    def _covers(
        self,
        key: str,
        cached: pd.DataFrame,
        coverage_start: str | None,
        coverage_end: str | None,
    ) -> bool:
        """Does the cached entry fully cover the requested ``[start, end]`` range?

        Uses sidecar metadata when present; otherwise infers the covered span from the
        frame's own time axis. When coverage can't be determined at all (no metadata and
        no recognisable date column) we warn and treat it as covered, to avoid hammering
        the network on payloads that simply have no time axis.
        """
        if coverage_start is None and coverage_end is None:
            return True

        meta = self._read_meta(key)
        cov_start = cov_end = None
        if meta is not None:
            cov_start = meta.get("coverage_start")
            cov_end = meta.get("coverage_end")
        if cov_start is None and cov_end is None:
            span = _frame_span(cached)
            if span is None:
                warnings.warn(
                    f"ParquetCache: no coverage metadata or date column for key "
                    f"{key!r}; cannot verify it covers the requested range — serving "
                    f"the cached entry as-is.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return True
            cov_start, cov_end = span[0].isoformat(), span[1].isoformat()

        if coverage_start is not None and cov_start is not None:
            if pd.Timestamp(cov_start) > pd.Timestamp(coverage_start):
                return False  # cached history starts too late -> truncated
        if coverage_end is not None and cov_end is not None:
            if pd.Timestamp(cov_end) < pd.Timestamp(coverage_end):
                return False  # cached history ends too early -> stale tail
        return True

    def _is_fresh(self, key: str, max_age: float | None) -> bool:
        if max_age is None:
            return True
        meta = self._read_meta(key)
        if meta is None or "saved_at" not in meta:
            # No timestamp recorded -> can't prove freshness; treat as stale so a TTL
            # request re-fetches rather than serving unknown-age data.
            return False
        return (time.time() - float(meta["saved_at"])) <= max_age

    def get_or_fetch(
        self,
        key: str,
        fetch: Callable[[], pd.DataFrame],
        refresh: bool = False,
        *,
        coverage_start: str | None = None,
        coverage_end: str | None = None,
        max_age: float | None = None,
    ) -> pd.DataFrame:
        """Return cached data for ``key`` or fetch & cache it.

        Backward compatible: calling with just ``key``/``fetch``/``refresh`` behaves
        exactly as before. The optional keyword args add safety:

        - ``coverage_start`` / ``coverage_end``: the date range the caller needs. If the
          cached entry does not fully cover it, the cache is bypassed and ``fetch`` is
          re-run (fixing the silent truncated-history bug).
        - ``max_age``: TTL in seconds. If the cached entry is older than this, re-fetch.
        """
        if not refresh:
            cached = self.load(key)
            if (
                cached is not None
                and self._is_fresh(key, max_age)
                and self._covers(key, cached, coverage_start, coverage_end)
            ):
                return cached
        df = fetch()
        self.save(
            key, df, coverage_start=coverage_start, coverage_end=coverage_end
        )
        return df
