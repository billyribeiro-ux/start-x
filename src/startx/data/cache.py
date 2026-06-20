"""Local Parquet cache so we never re-pull the same history from FMP."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import pandas as pd

_SAFE = re.compile(r"[^A-Za-z0-9_-]")


class ParquetCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        parts = [_SAFE.sub("_", p) for p in key.split("/") if p]
        path = self.dir.joinpath(*parts).with_suffix(".parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def load(self, key: str) -> pd.DataFrame | None:
        path = self._path(key)
        return pd.read_parquet(path) if path.exists() else None

    def save(self, key: str, df: pd.DataFrame) -> None:
        if df is not None:
            df.to_parquet(self._path(key), index=False)

    def get_or_fetch(
        self, key: str, fetch: Callable[[], pd.DataFrame], refresh: bool = False
    ) -> pd.DataFrame:
        if not refresh:
            cached = self.load(key)
            if cached is not None:
                return cached
        df = fetch()
        self.save(key, df)
        return df
