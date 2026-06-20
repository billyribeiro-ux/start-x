"""Lightweight model registry: persist fitted models + JSON metadata to ``artifacts/``.

Each ``save_model`` call writes two timestamped files under ``<dir>/<name>/``:

* ``<name>-<timestamp>.joblib`` — the pickled estimator;
* ``<name>-<timestamp>.json``   — metadata (user ``meta`` plus provenance: name, timestamp,
  model class, joblib filename).

:func:`load_model` / :func:`latest` resolve either an exact ``stamp`` or the most recent
artifact for ``name``. ``artifacts/`` is gitignored.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

_DEFAULT_DIR = "artifacts"


def _stamp() -> str:
    """UTC timestamp suitable for a filename (sortable, lexicographic == chronological)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def save_model(model: Any, meta: dict, name: str, dir: str | Path = _DEFAULT_DIR) -> Path:
    """Persist ``model`` + ``meta`` under ``<dir>/<name>/``; return the ``.joblib`` path.

    ``meta`` is shallow-copied and enriched with provenance keys (``name``, ``saved_at``,
    ``model_class``, ``artifact``). Non-JSON-serializable values are stringified.
    """
    base = Path(dir) / name
    base.mkdir(parents=True, exist_ok=True)

    stamp = _stamp()
    model_path = base / f"{name}-{stamp}.joblib"
    meta_path = base / f"{name}-{stamp}.json"

    joblib.dump(model, model_path)

    full_meta = dict(meta)
    full_meta.update(
        {
            "name": name,
            "saved_at": stamp,
            "model_class": type(model).__name__,
            "artifact": model_path.name,
        }
    )
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(full_meta, fh, indent=2, default=str)

    return model_path


def _meta_files(name: str, dir: str | Path) -> list[Path]:
    """All metadata JSON files for ``name`` sorted ascending (oldest first)."""
    base = Path(dir) / name
    if not base.exists():
        return []
    return sorted(base.glob(f"{name}-*.json"))


def latest(name: str, dir: str | Path = _DEFAULT_DIR) -> Path | None:
    """Path to the most recent ``.joblib`` artifact for ``name`` (or ``None`` if none)."""
    metas = _meta_files(name, dir)
    if not metas:
        return None
    newest = metas[-1]
    return newest.with_suffix(".joblib")


def load_model(
    name: str, dir: str | Path = _DEFAULT_DIR, *, stamp: str | None = None
) -> tuple[Any, dict]:
    """Load ``(model, meta)`` for ``name``; newest by default or the exact ``stamp``.

    Raises ``FileNotFoundError`` when no matching artifact exists.
    """
    base = Path(dir) / name
    if stamp is not None:
        model_path = base / f"{name}-{stamp}.joblib"
        meta_path = base / f"{name}-{stamp}.json"
    else:
        model_path = latest(name, dir)
        if model_path is None:
            raise FileNotFoundError(f"no saved model named {name!r} under {dir}")
        meta_path = model_path.with_suffix(".json")

    if not model_path.exists():
        raise FileNotFoundError(f"model artifact missing: {model_path}")

    model = joblib.load(model_path)
    meta: dict = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
    return model, meta
