"""Typed qedge wrappers over the proven ``startx.labeling`` / ``startx.learning``.

These are *thin* delegating wrappers: they inject the qedge config presets
(``cfg.labeling.{short,long}_*``) and forward to the already-validated startx
implementations. No methodology is re-implemented here — the contract reuses the
startx modules verbatim and merely binds the qedge horizon/barrier presets so the
short-swing book (10-day cap) and long-swing book (~63-day cap) are produced from
a single source of truth.

The underlying startx names are re-exported so callers can reach the raw
primitives (e.g. ``daily_vol``) without importing startx directly.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd

from qedge.config import get_config
from startx.labeling.triple_barrier import (
    LABEL_COLUMNS,
    daily_vol,
    triple_barrier_labels,
)
from startx.labeling.weights import (
    average_uniqueness,
    num_concurrent_events,
    sample_weights,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from startx.learning.metalabel import apply_meta_filter, walk_forward_metalabel

__all__ = [
    "LABEL_COLUMNS",
    "apply_meta_filter",
    "average_uniqueness",
    "daily_vol",
    "label_long",
    "label_short",
    "metalabel",
    "num_concurrent_events",
    "sample_weights",
    "triple_barrier_labels",
    "uniqueness_weights",
    "walk_forward_metalabel",
]


def label_short(prices: pd.DataFrame, *, vol: pd.Series | None = None) -> pd.DataFrame:
    """Triple-barrier labels for the SHORT-swing book (1-10 day horizon).

    Binds the ``cfg.labeling.short_*`` presets (hard 10-day vertical-barrier cap,
    1.5-ATR-style profit/stop widths) and delegates to
    :func:`startx.labeling.triple_barrier.triple_barrier_labels`.

    Parameters
    ----------
    prices:
        OHLCV frame with at least ``date, high, low, close`` columns.
    vol:
        Optional precomputed volatility series (positionally aligned to
        ``prices``); when ``None`` it is computed from ``close`` using the
        configured ``short_vol_span``.

    Returns
    -------
    pd.DataFrame
        Schema :data:`startx.labeling.triple_barrier.LABEL_COLUMNS`.
    """
    labeling = get_config().labeling
    return triple_barrier_labels(
        prices,
        horizon_days=labeling.short_horizon_days,
        pt_mult=labeling.short_pt_mult,
        sl_mult=labeling.short_sl_mult,
        vol_span=labeling.short_vol_span,
        vol=vol,
    )


def label_long(prices: pd.DataFrame, *, vol: pd.Series | None = None) -> pd.DataFrame:
    """Triple-barrier labels for the LONG-swing book (weeks to ~3 months).

    Binds the ``cfg.labeling.long_*`` presets (~63-day vertical-barrier cap,
    wider 2.5-ATR-style profit/stop widths) and delegates to
    :func:`startx.labeling.triple_barrier.triple_barrier_labels`.

    Parameters mirror :func:`label_short`.
    """
    labeling = get_config().labeling
    return triple_barrier_labels(
        prices,
        horizon_days=labeling.long_horizon_days,
        pt_mult=labeling.long_pt_mult,
        sl_mult=labeling.long_sl_mult,
        vol_span=labeling.long_vol_span,
        vol=vol,
    )


def uniqueness_weights(
    bar_dates: pd.DatetimeIndex,
    t1: pd.Series,
    ret: pd.Series | None = None,
    time_decay: float | None = None,
) -> pd.Series:
    """Per-label sample weights for overlapping triple-barrier labels.

    Thin delegation to :func:`startx.labeling.weights.sample_weights` — combines
    average uniqueness (overlap down-weighting), optional return attribution and
    optional linear time-decay, normalised to mean 1.0.
    """
    return sample_weights(bar_dates, t1, ret=ret, time_decay=time_decay)


def metalabel(
    features: pd.DataFrame,
    labels: pd.Series,
    entry_times: pd.Series,
    t1: pd.Series,
    model_factory: Callable[[], Any],
    *,
    train_min: int | None = None,
    test_span: int | None = None,
) -> pd.DataFrame:
    """Walk-forward, point-in-time-clean meta-labelling (take/skip filter).

    Delegates to :func:`startx.learning.metalabel.walk_forward_metalabel`. When
    ``train_min`` / ``test_span`` are omitted they default to the configured
    walk-forward presets (``cfg.validation.walkforward_min_train`` and
    ``walkforward_test_span``) so the multiple-testing/firewall sizing is bound to
    the same single source of truth as the rest of validation.

    Returns the OOS ``meta_prob_win`` frame
    (``[entry_time, meta_prob_win, y_true, y_pred]``).
    """
    # Lazy import: startx.learning.__init__ pulls in heavy ML deps (sklearn /
    # lightgbm); deferring the import keeps the numpy/pandas-only labeling
    # wrappers importable in a bare environment, and the metalabel path only
    # runs when an estimator (which needs those deps anyway) is supplied.
    from startx.learning.metalabel import walk_forward_metalabel

    validation = get_config().validation
    resolved_train_min = (
        validation.walkforward_min_train if train_min is None else train_min
    )
    resolved_test_span = (
        validation.walkforward_test_span if test_span is None else test_span
    )
    return walk_forward_metalabel(
        features,
        labels,
        entry_times,
        t1,
        model_factory,
        train_min=resolved_train_min,
        test_span=resolved_test_span,
    )


def __getattr__(name: str) -> object:
    """Lazily re-export the startx meta-labelling primitives (PEP 562).

    ``walk_forward_metalabel`` and ``apply_meta_filter`` live behind
    ``startx.learning``, whose package import drags in heavy ML dependencies.
    Resolving them on first attribute access keeps ``import qedge.labeling`` cheap
    and dependency-light while still exposing the names for re-export.
    """
    if name in ("walk_forward_metalabel", "apply_meta_filter"):
        from startx.learning import metalabel as _metalabel

        return getattr(_metalabel, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
