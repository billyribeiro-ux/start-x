"""Sample weighting for overlapping (non-IID) triple-barrier labels.

Triple-barrier labels span ``[t, t1]`` and overlap in time, so they are *not*
independent: bars that contribute to many concurrent labels carry less unique
information. Following López de Prado (*Advances in Financial ML*, Ch. 4) we:

1. count concurrent labels per bar (:func:`num_concurrent_events`),
2. average ``1 / concurrency`` over each label's life (:func:`average_uniqueness`),
3. derive sample weights from uniqueness, optionally scaled by return attribution
   and a linear time-decay (:func:`sample_weights`).

Each label is identified by its index in ``t1`` (the index is the entry time ``t``,
the value is the exit time ``t1``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _entry_times(t1: pd.Series) -> pd.DatetimeIndex:
    """Entry timestamps for each label (the index of ``t1``)."""
    return pd.DatetimeIndex(pd.to_datetime(t1.index))


def num_concurrent_events(bar_dates: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """Number of labels live on each bar in ``bar_dates``.

    A label indexed by ``t`` with exit ``t1`` is "live" on every bar in ``[t, t1]``
    (inclusive). Returns a series indexed by ``bar_dates`` of integer counts.
    """
    bar_dates = pd.DatetimeIndex(pd.to_datetime(bar_dates))
    t1 = pd.to_datetime(t1)
    entries = _entry_times(t1)
    count = pd.Series(0, index=bar_dates, dtype="int64")
    for t_in, t_out in zip(entries, t1.to_numpy()):
        # Mark every bar between entry and exit (inclusive) as covered by this label.
        count.loc[t_in:pd.Timestamp(t_out)] += 1
    return count


def average_uniqueness(bar_dates: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """Average uniqueness of each label = mean of ``1/concurrency`` over its life.

    Returns a series indexed like ``t1`` (i.e. by entry time ``t``) with values in
    ``(0, 1]``; 1.0 means the label never overlapped another, ~0.5 means it shared
    every bar with exactly one other label, etc.
    """
    bar_dates = pd.DatetimeIndex(pd.to_datetime(bar_dates))
    t1 = pd.to_datetime(t1)
    conc = num_concurrent_events(bar_dates, t1)
    entries = _entry_times(t1)
    inv = 1.0 / conc.replace(0, np.nan)
    avg = pd.Series(index=t1.index, dtype="float64")
    for label_id, t_in, t_out in zip(t1.index, entries, t1.to_numpy()):
        avg.loc[label_id] = inv.loc[t_in:pd.Timestamp(t_out)].mean()
    return avg


def sample_weights(
    bar_dates: pd.DatetimeIndex,
    t1: pd.Series,
    ret: pd.Series | None = None,
    time_decay: float | None = None,
) -> pd.Series:
    """Per-label sample weights for fitting on overlapping labels.

    Combines:

    * **uniqueness** — average uniqueness over each label's life (down-weights
      labels living on heavily-overlapped bars);
    * **return attribution** (optional) — when ``ret`` is given, weights scale with
      ``|ret|`` so larger realized moves count more (magnitude only, sign-agnostic);
    * **linear time-decay** (optional) — when ``time_decay`` is in ``(0, 1]``, older
      labels are linearly down-weighted; ``decay`` is the weight floor applied to the
      oldest label (``1.0`` = no decay). Decay is applied over cumulative uniqueness
      so it tracks information content, not raw calendar time.

    Weights are normalized to **mean 1.0**. Returns a series indexed like ``t1``.
    """
    avg_u = average_uniqueness(bar_dates, t1)
    w = avg_u.copy()

    if ret is not None:
        ret = ret.reindex(avg_u.index)
        w = avg_u * ret.abs()

    if time_decay is not None:
        if not (0.0 < time_decay <= 1.0):
            raise ValueError(f"time_decay must be in (0, 1], got {time_decay}")
        # Order labels by entry time, then build a linear decay over cumulative
        # uniqueness so total decayed weight tracks unique information, per LdP Ch.4.
        order = pd.DatetimeIndex(pd.to_datetime(avg_u.index)).argsort()
        ordered_ids = avg_u.index[order]
        cum = avg_u.loc[ordered_ids].cumsum()
        total = cum.iloc[-1] if len(cum) else 0.0
        if total > 0:
            if time_decay >= 0:
                slope = (1.0 - time_decay) / total
            else:  # documented LdP extension: negative kills oldest labels entirely
                slope = 1.0 / ((time_decay + 1.0) * total)
            const = 1.0 - slope * total
            decay = const + slope * cum
            decay[decay < 0] = 0.0
            w = w * decay.reindex(w.index)

    total_w = w.sum()
    if total_w <= 0 or not np.isfinite(total_w):
        # Degenerate (all-zero) weights: fall back to uniform.
        return pd.Series(1.0, index=avg_u.index)
    return w * (len(w) / total_w)
