"""Loss autopsy — turn each trade into entry-day microstructure forensics, then rank what
separates winners from losers.

Two functions:

* :func:`autopsy_trades` — for every trade, attach the *entry-day* daily forensics from
  :func:`startx.events.reversals.characterize_reversal` (rvol, ATR, ATR-expansion, gap, close
  location value, abnormal-return z, and a one-hot of the candle archetype), plus — when an
  intraday client is supplied — the close-vs-POC / close-vs-VWAP / cumulative-delta relations.
  It adds ``is_win = (ret_net > 0)`` so downstream stages have a clean label.

* :func:`win_loss_signature` — rank every numeric forensic feature by how strongly it separates
  wins from losses (single-feature AUC and mean lift). This is the automated "why losses
  happen" table.

LEAKAGE DISCIPLINE (this is THE most leakage-sensitive module):
  Forensic features for a trade are computed from the entry day (or earlier) ONLY. We pass the
  ``entry_date`` as the "reversal date" to :func:`characterize_reversal`, which dissects that one
  day using only that day's and trailing data (trailing-20 volume mean, Wilder ATR of the past,
  prev-close gap, abnormal-return z). We NEVER feed the trade's exit price, exit date or its
  outcome (``ret_net`` / ``is_win``) in as a feature. ``is_win`` is the *label*, kept in its own
  column and excluded from the ranked feature space.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..events.reversals import characterize_reversal

#: Daily forensic keys lifted from ``characterize_reversal``'s ``daily`` block. ``candle`` is the
#: categorical archetype and is expanded into one-hot ``candle_<label>`` columns separately.
_DAILY_NUMERIC = (
    "rvol_20",
    "atr_14",
    "atr_expansion",
    "gap_pct",
    "close_location_value",
    "ar_z",
)
#: Intraday relations (only populated when ``intraday`` and a ``client`` are supplied).
_INTRADAY_NUMERIC = (
    "reversal_vs_poc",
    "reversal_vs_vwap",
    "cumulative_delta",
)
#: Columns that are trade identity / outcome, NOT forensic features — never ranked or modelled.
_OUTCOME_COLS = ("symbol", "entry_date", "exit_date", "ret_net", "ret_gross", "is_win", "t1")
#: Prefix marking the one-hot candle-archetype flags.
_CANDLE_PREFIX = "candle_"
#: The candle archetype labels :func:`startx.events.reversals._candle_label` can emit (plus the
#: ``unknown`` fallback used when a bar can't be characterized). A ``candle_<name>`` column whose
#: ``<name>`` is in this set is a genuine one-hot flag from this module. Any OTHER ``candle_*``
#: column is trusted ONLY if it is actually 0/1-valued — see :func:`forensic_feature_columns`.
_CANDLE_ARCHETYPES = frozenset({
    "flat", "doji", "hammer", "hanging_man", "shooting_star", "inverted_hammer",
    "bullish_marubozu_gap_up", "strong_bullish", "bearish_marubozu_gap_down",
    "strong_bearish", "bullish", "bearish", "unknown",
})
#: Carried *pre-entry* signal columns that ARE legitimate features (the primary model's own signal
#: known at decision time). Extend this when a new primary signal is added. Anything NOT here and
#: not a forensic/candle column is excluded — see :func:`forensic_feature_columns`.
_SIGNAL_COLS = ("prob_up", "prob_win", "meta_prob_win", "ibs_entry", "rsi2_entry", "vix_level",
                "vix_pctile")


def _nan_daily() -> dict[str, float]:
    """All-NaN daily forensic block (used when a trade's prices/date are missing)."""
    out: dict[str, float] = {k: float("nan") for k in _DAILY_NUMERIC}
    out["candle"] = "unknown"
    return out


def autopsy_trades(
    trades: pd.DataFrame,
    prices_by: dict[str, pd.DataFrame],
    *,
    client=None,
    cache=None,
    intraday: bool = False,
) -> pd.DataFrame:
    """Attach entry-day forensic columns to each trade and add the ``is_win`` label.

    Parameters
    ----------
    trades:
        One row per trade. Must contain ``symbol``, ``entry_date`` and ``ret_net``. Any other
        columns (``exit_date``, ``ret_gross``, ``t1``, a primary-model probability, …) are
        carried through untouched.
    prices_by:
        ``{symbol -> enriched daily price frame}`` (as produced by ``get_prices`` /
        ``compute_signals``); the forensics for a trade come from its symbol's frame.
    client, cache, intraday:
        When ``intraday`` is true and a ``client`` is supplied, the entry-day intraday relations
        (close-vs-POC, close-vs-VWAP, cumulative delta) are added on top of the daily block.

    Returns
    -------
    pd.DataFrame
        ``trades`` plus the numeric daily forensic columns, one-hot ``candle_*`` flags, optional
        intraday columns, and ``is_win = (ret_net > 0).astype(int)``. Robust to missing
        forensics: any trade whose prices/date can't be characterized gets NaN features (and a
        ``candle_unknown`` flag) rather than raising.
    """
    if trades is None or trades.empty:
        out = pd.DataFrame(trades).copy() if trades is not None else pd.DataFrame()
        out["is_win"] = pd.Series(dtype=int)
        return out

    work = trades.reset_index(drop=True).copy()
    daily_rows: list[dict] = []
    intraday_rows: list[dict] = []

    for _, trade in work.iterrows():
        symbol = trade.get("symbol")
        entry_date = trade.get("entry_date")
        prices = prices_by.get(symbol) if symbol is not None else None

        daily: dict
        intra: dict = {k: float("nan") for k in _INTRADAY_NUMERIC}
        if prices is None or prices.empty or pd.isna(entry_date):
            daily = _nan_daily()
        else:
            try:
                # PIT firewall: characterize the ENTRY day only. characterize_reversal uses the
                # entry day's own bar plus trailing context (ATR/volume/gap/ar_z) — never the
                # trade's exit or outcome. fmp_symbol gates intraday lookups to the right name.
                char = characterize_reversal(
                    prices,
                    pd.Timestamp(entry_date),
                    client=client if intraday else None,
                    cache=cache,
                    fmp_symbol=str(symbol) if (intraday and symbol is not None) else None,
                    intraday=bool(intraday and client is not None),
                )
                daily = char.get("daily", _nan_daily())
                if intraday:
                    block = char.get("intraday", {})
                    for k in _INTRADAY_NUMERIC:
                        intra[k] = float(block.get(k, np.nan))
            except Exception:  # noqa: BLE001 — never let one bad trade abort the autopsy
                daily = _nan_daily()

        row = {k: float(daily.get(k, np.nan)) for k in _DAILY_NUMERIC}
        row["candle"] = str(daily.get("candle", "unknown"))
        daily_rows.append(row)
        intraday_rows.append(intra)

    forensic = pd.DataFrame(daily_rows, index=work.index)
    # One-hot the candle archetype: candle_<label> presence flags (int 0/1).
    candle = forensic.pop("candle")
    onehot = pd.get_dummies(candle, prefix=_CANDLE_PREFIX.rstrip("_")).astype(int)

    out = pd.concat([work, forensic, onehot], axis=1)
    if intraday:
        out = pd.concat([out, pd.DataFrame(intraday_rows, index=work.index)], axis=1)

    # The LABEL — kept separate from the feature space, derived from net return only.
    ret_net = pd.to_numeric(out["ret_net"], errors="coerce")
    out["is_win"] = (ret_net > 0).astype(int)
    return out


def _is_binary_flag(series: pd.Series) -> bool:
    """True iff ``series`` is a genuine 0/1 one-hot flag (no other numeric values).

    A real ``candle_*`` one-hot only ever holds 0 or 1. Anything else — a continuous
    return, a price level, a probability — must NOT slip through the ``candle_`` prefix
    on the allow-list. NaNs are ignored (a missing flag is still a flag); an all-NaN or
    empty column is rejected (nothing proves it is binary).
    """
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return False
    return bool(np.isin(vals, (0.0, 1.0)).all())


def forensic_feature_columns(autopsy: pd.DataFrame) -> list[str]:
    """Eligible forensic feature space for :func:`win_loss_signature` and the meta-model — an
    ALLOW-list, not a deny-list.

    A deny-list ("every numeric column except a few ids") silently LEAKS the moment a strategy's
    trade frame carries an outcome/level column the list doesn't name — e.g. ``ret``, ``exit_price``,
    or the absolute ``entry_price``/``target``/``stop`` levels. Feeding those trains the meta-model
    on the answer (OOS AUC -> 1.0, the "100% win" overfit trap this project exists to reject). So we
    allow ONLY: the entry-day forensic families this module attaches (scale-free by design), the
    one-hot ``candle_*`` archetypes, and explicitly-registered pre-entry signals (``_SIGNAL_COLS``).

    The ``candle_`` prefix is NOT a blank cheque: a numeric ``candle_LEAKED_RETURN`` would
    otherwise sail through. A ``candle_*`` column is accepted only when it is a real one-hot flag —
    its archetype name is one this module emits (``_CANDLE_ARCHETYPES``) OR its values are strictly
    0/1 (:func:`_is_binary_flag`). A continuous column wearing the ``candle_`` prefix is rejected.
    """
    allowed = set(_DAILY_NUMERIC) | set(_INTRADAY_NUMERIC) | set(_SIGNAL_COLS)
    cols: list[str] = []
    for c in autopsy.columns:
        if c in _OUTCOME_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(autopsy[c]):
            continue
        if c in allowed:
            cols.append(c)
        elif c.startswith(_CANDLE_PREFIX):
            archetype = c[len(_CANDLE_PREFIX):]
            # A real candle one-hot: a known archetype name AND/OR strictly 0/1 values.
            if archetype in _CANDLE_ARCHETYPES or _is_binary_flag(autopsy[c]):
                cols.append(c)
    return cols


def _auc(values: np.ndarray, is_win: np.ndarray) -> float:
    """Single-feature AUC: P(feature higher on a win than on a loss).

    Rank-based Mann-Whitney estimator with ties counted as 0.5 (monotone-invariant, model-free).
    0.5 == no separation; far from 0.5 == strong single-feature signal. Mirrors the estimator in
    :mod:`startx.events.evidence` so the autopsy ranking reads on the same scale.
    """
    pos = values[is_win]
    neg = values[~is_win]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    # Average ranks within tie groups so ties contribute exactly 0.5.
    s = np.sort(values)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            mask = values == s[i]
            ranks[mask] = ranks[mask].mean()
        i = j + 1
    sum_pos = ranks[is_win].sum()
    return float((sum_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def win_loss_signature(autopsy: pd.DataFrame) -> pd.DataFrame:
    """Rank each numeric forensic feature by how strongly it separates wins from losses.

    Columns: ``[feature, mean_at_win, mean_at_loss, single_feature_auc, lift]``, sorted by
    ``|single_feature_auc - 0.5|`` descending — the strongest separators (the clearest
    fingerprints of a losing setup) first. ``lift`` is ``mean_at_win / mean_at_loss``.

    This is the automated "why losses happen" — purely descriptive (it does not gate any trade),
    so it may look at every resolved trade in ``autopsy`` at once.
    """
    cols = ["feature", "mean_at_win", "mean_at_loss", "single_feature_auc", "lift"]
    if autopsy is None or autopsy.empty or "is_win" not in autopsy.columns:
        return pd.DataFrame(columns=cols)

    is_win_all = pd.to_numeric(autopsy["is_win"], errors="coerce").to_numpy()
    features = forensic_feature_columns(autopsy)
    rows: list[dict] = []
    for feat in features:
        col = pd.to_numeric(autopsy[feat], errors="coerce").to_numpy(dtype=float)
        valid = ~np.isnan(col) & ~np.isnan(is_win_all)
        v = col[valid]
        w = is_win_all[valid] > 0
        if v.size < 3 or w.sum() == 0 or (~w).sum() == 0:
            continue
        mean_at_win = float(v[w].mean())
        mean_at_loss = float(v[~w].mean())
        auc = _auc(v, w)
        lift = float(mean_at_win / mean_at_loss) if mean_at_loss != 0 else float("nan")
        rows.append({
            "feature": feat,
            "mean_at_win": mean_at_win,
            "mean_at_loss": mean_at_loss,
            "single_feature_auc": round(auc, 4) if auc == auc else auc,
            "lift": round(lift, 4) if lift == lift else lift,
            "_strength": abs(auc - 0.5) if auc == auc else -1.0,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=cols)
    out = out.sort_values("_strength", ascending=False).reset_index(drop=True)
    return out.drop(columns=["_strength"])
