"""Reproducibility layer: deterministic seeds, data-snapshot hashing, run logging.

The contract requires reproducible runs with logged configs and hashed data
snapshots. Everything here is pure/deterministic: the same inputs always produce
the same hashes and the same seeded RNG stream.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def set_seeds(seed: int) -> None:
    """Seed Python and NumPy RNGs for deterministic behaviour.

    Hashing of strings is also pinned via ``PYTHONHASHSEED`` for child processes
    (it cannot affect the already-running interpreter, hence set as env only).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # Seed the legacy global RNG too: some third-party libs (sklearn estimators
    # without an explicit random_state) still draw from it. New qedge code should
    # use rng() Generators instead.
    np.random.seed(seed)  # noqa: NPY002


def rng(seed: int) -> np.random.Generator:
    """Return a fresh, independent NumPy ``Generator`` for the given seed.

    Prefer this over the global RNG for feature/synthetic code so that parallel
    components do not share mutable global state.
    """
    return np.random.default_rng(seed)


def hash_frame(frame: pd.DataFrame) -> str:
    """Return a stable sha256 over a DataFrame's content (order-sensitive).

    The hash is invariant to the in-memory block layout but sensitive to values,
    column names, dtypes, and row/column order — so two logically identical
    snapshots hash equal while any data change is detected.
    """
    hasher = hashlib.sha256()
    # Column names + dtypes first, so a dtype or rename is caught.
    schema = json.dumps(
        [(str(c), str(frame[c].dtype)) for c in frame.columns],
        separators=(",", ":"),
    )
    hasher.update(schema.encode("utf-8"))
    # Row index, then the raw bytes of the values in column order.
    index_values = pd.util.hash_pandas_object(frame.index, index=False).to_numpy()
    hasher.update(index_values.tobytes())
    for col in frame.columns:
        col_hash = pd.util.hash_pandas_object(frame[col], index=False).to_numpy()
        hasher.update(col_hash.tobytes())
    return hasher.hexdigest()


def hash_bytes(payload: bytes) -> str:
    """Return a sha256 hex digest of arbitrary bytes."""
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class RunRecord:
    """An immutable, serialisable record of one scanner run."""

    run_id: str
    created_at: str
    config: dict[str, Any]
    data_hashes: dict[str, str]
    metrics: dict[str, Any]

    def to_json(self) -> str:
        """Serialise the record to indented, deterministic JSON."""
        return json.dumps(
            {
                "run_id": self.run_id,
                "created_at": self.created_at,
                "config": self.config,
                "data_hashes": self.data_hashes,
                "metrics": self.metrics,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )


def make_run_record(
    config: dict[str, Any],
    data_hashes: dict[str, str],
    metrics: dict[str, Any],
    *,
    now: datetime | None = None,
) -> RunRecord:
    """Build a :class:`RunRecord`; ``run_id`` derives from config+data hashes.

    The ``run_id`` is deterministic in the config and data snapshot (not the
    wall clock), so re-running identical inputs yields the same id — the audit
    trail dedupes naturally.
    """
    stamp = (now or datetime.now(UTC)).isoformat()
    fingerprint = hash_bytes(
        json.dumps(
            {"config": config, "data_hashes": data_hashes},
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    )
    return RunRecord(
        run_id=fingerprint[:16],
        created_at=stamp,
        config=config,
        data_hashes=data_hashes,
        metrics=metrics,
    )


def write_run_record(record: RunRecord, output_dir: str | Path) -> Path:
    """Persist a run record as ``<output_dir>/<run_id>.json`` and return its path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{record.run_id}.json"
    path.write_text(record.to_json(), encoding="utf-8")
    return path
