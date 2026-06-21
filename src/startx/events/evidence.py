"""Evidence Miner — the cross-year "main scanner".

Across many tickers and years this module snapshots, for every statistically significant
move, the conditions that *preceded* it and then ranks — side by side, up vs down — which
features and catalysts matched the most.

POINT-IN-TIME RULE (the whole point of this file): a move happens on day ``t``. The conditions
that preceded it are the feature row at the *prior* trading day ``t-1`` — information available
*before* the move. We merge each move to the feature matrix shifted by one trading day, so day
``t``'s own features can never be used to "explain" day ``t``'s move. Catalysts are attributed
only from inside the causal lookback window ``[t - lookback trading days, t's close]`` (handled by
``attribute_event``): a catalyst time-stamped *after* day ``t``'s cash close (post-close news,
analyst actions, AMC earnings) is gated out and pushed to ``t+1``, so the catalyst flags are also
strictly information available at or before ``t``'s close.

Ranking quantifies, for each numeric feature, how strongly its level at ``t-1`` separates
up-moves from down-moves (point-biserial correlation, single-feature AUC, mean lift), so the
output reads as "what matched the most" for up versus down.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..data.cache import ParquetCache
from ..data.catalysts import load_catalysts
from ..data.prices import get_prices
from ..data.universe import Universe, load_universe
from ..features.assemble import build_feature_matrix
from ..fmp.client import FMPClient
from ..settings import Settings, get_settings
from .attribute import attribute_event
from .detect import compute_signals, detect_events

#: Identity / non-feature columns carried alongside each move; never ranked as conditions.
_ID_COLS = ("date", "symbol", "direction", "abs_ar_pct", "ar_pct", "ar_z", "vol_z")
#: Non-numeric metadata that ``build_feature_matrix`` emits but which is not a numeric feature.
_NON_FEATURE = ("symbol", "sector", "date")
#: Prefix marking the attributed-catalyst presence flags on each evidence row.
_CAT_PREFIX = "cat_"


def _feature_columns(evidence: pd.DataFrame) -> list[str]:
    """Numeric feature columns eligible for ranking (PIT snapshot, not id/catalyst/meta)."""
    skip = set(_ID_COLS) | set(_NON_FEATURE)
    cols: list[str] = []
    for c in evidence.columns:
        if c in skip or c.startswith(_CAT_PREFIX):
            continue
        if pd.api.types.is_numeric_dtype(evidence[c]):
            cols.append(c)
    return cols


def collect_move_evidence(
    tickers: Sequence[str] | Iterable[str],
    start: str,
    end: str,
    *,
    client: FMPClient,
    cache: ParquetCache | None = None,
    universe: Universe | None = None,
    settings: Settings | None = None,
    ar_threshold: float = 2.5,
    lookback_days: int = 2,
) -> pd.DataFrame:
    """One row per significant move across all ``tickers``/years, with PRIOR-day conditions.

    For each ticker: pull prices -> ``compute_signals`` -> ``detect_events`` to find the moves,
    build the PIT feature matrix, then merge each move (day ``t``) to the feature snapshot at the
    prior trading day (``t-1``) via a one-row shift. The attributed catalyst types present in the
    pre-move window are recorded as ``cat_<type>`` presence flags plus a ``top_catalyst`` label.

    Returned columns: ``date``, ``symbol``, ``direction``, ``abs_ar_pct`` (plus ``ar_pct``,
    ``ar_z``, ``vol_z`` context), every numeric PIT feature taken at ``t-1``, the ``cat_*``
    catalyst-presence flags and ``top_catalyst``.
    """
    settings = settings or get_settings()
    cache = cache or ParquetCache(settings.cache_dir)
    universe = universe or load_universe()

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        frame = _collect_one(
            ticker, start, end, client=client, cache=cache, universe=universe,
            settings=settings, ar_threshold=ar_threshold, lookback_days=lookback_days,
        )
        if frame is not None and not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=list(_ID_COLS) + ["top_catalyst"])
    out = pd.concat(frames, ignore_index=True, sort=False)
    # Stable, readable column order: ids first, catalyst flags last, features in between.
    cat_cols = sorted(c for c in out.columns if c.startswith(_CAT_PREFIX))
    feat_cols = [c for c in out.columns
                 if c not in _ID_COLS and c not in cat_cols and c != "top_catalyst"]
    ordered = [c for c in _ID_COLS if c in out.columns] + feat_cols + cat_cols + ["top_catalyst"]
    out = out[[c for c in ordered if c in out.columns]]
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def _collect_one(
    ticker: str,
    start: str,
    end: str,
    *,
    client: FMPClient,
    cache: ParquetCache,
    universe: Universe,
    settings: Settings,
    ar_threshold: float,
    lookback_days: int,
) -> pd.DataFrame | None:
    """Evidence rows for a single ticker (or ``None`` if it has no usable data/moves)."""
    spec = universe.spec(ticker)
    prices = get_prices(client, cache, spec.fmp, settings.history_start)
    if prices.empty:
        return None

    market = None
    if spec.is_stock:
        bench = get_prices(client, cache, universe.benchmark, settings.history_start)
        market = bench if not bench.empty else None

    signals = compute_signals(prices, market)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    win = signals[(signals["date"] >= start_ts) & (signals["date"] <= end_ts)]
    events = detect_events(win, ar_threshold=ar_threshold)
    if events.empty:
        return None

    matrix = build_feature_matrix(
        ticker, start, end, client=client, cache=cache, universe=universe, settings=settings,
    )
    if matrix.empty:
        return None
    matrix = matrix.sort_values("date").reset_index(drop=True)
    matrix["date"] = pd.to_datetime(matrix["date"])

    prior = _prior_day_snapshot(matrix)
    if prior.empty:
        return None

    events = events.copy()
    events["date"] = pd.to_datetime(events["date"])
    # Keep ONLY move-identity/context columns from the event row. The other event columns
    # (ret_pct, gap, close, volume, beta, fwd_ret_*) describe day t's move itself — using them
    # as "preceding conditions" would violate PIT, so they never enter the ranked feature space.
    keep = [c for c in ("date", "direction", "abs_ar_pct", "ar_pct", "ar_z", "vol_z")
            if c in events.columns]
    events = events[keep]
    # Merge each move (day t) to the conditions snapshot from the prior trading day (t-1).
    merged = events.merge(prior, on="date", how="inner", suffixes=("", "_feat"))
    if merged.empty:
        return None

    merged.insert(1, "symbol", ticker)
    catalysts = load_catalysts(client, cache, spec, start, end)
    merged = _attach_catalysts(merged, catalysts, lookback_days)
    return merged


def _prior_day_snapshot(matrix: pd.DataFrame) -> pd.DataFrame:
    """Align each trading day to the feature row of the PRIOR trading day (one-row shift).

    The result is keyed on the *move* date ``t`` but every feature value comes from ``t-1``.
    """
    # Drop id/meta and any forward-looking column defensively (features must be trailing/PIT).
    feat_cols = [c for c in matrix.columns
                 if c not in ("date", "symbol", "sector") and not c.startswith("fwd_")]
    snap = matrix[["date"] + feat_cols].copy()
    # Shift features forward by one trading row: row i carries the features observed at row i-1.
    shifted = snap[feat_cols].shift(1)
    shifted.insert(0, "date", snap["date"].to_numpy())
    return shifted.iloc[1:].reset_index(drop=True)  # first row has no prior day


def _attach_catalysts(
    moves: pd.DataFrame, catalysts: dict[str, pd.DataFrame], lookback_days: int
) -> pd.DataFrame:
    """Add ``cat_<type>`` presence flags + a ``top_catalyst`` label per move (PIT-attributed).

    Attribution runs through ``attribute_event``, which gates out any catalyst stamped after the
    move-day's close (advancing it to ``t+1``), so every flag here reflects only information public
    at or before day ``t``'s close — never a post-close catalyst leaked back onto the move.
    """
    flags: list[dict[str, int]] = []
    tops: list[str] = []
    for _, row in moves.iterrows():
        causes, top, _conf = attribute_event(row, catalysts, lookback_days=lookback_days)
        present = {f"{_CAT_PREFIX}{c['type']}": 1 for c in causes}
        flags.append(present)
        tops.append(top["type"] if top else "none")
    flag_df = pd.DataFrame(flags).fillna(0).astype(int)
    flag_df.index = moves.index
    out = pd.concat([moves, flag_df], axis=1)
    out["top_catalyst"] = tops
    return out


def _point_biserial(values: np.ndarray, is_up: np.ndarray) -> float:
    """Point-biserial correlation between a numeric feature and the up/down (1/0) label."""
    if values.size < 3:
        return float("nan")
    if np.nanstd(values) == 0:
        return 0.0
    up = values[is_up]
    down = values[~is_up]
    if up.size == 0 or down.size == 0:
        return float("nan")
    n = values.size
    sd = np.nanstd(values)
    if sd == 0:
        return 0.0
    return float((np.nanmean(up) - np.nanmean(down)) / sd
                 * np.sqrt(up.size * down.size / (n * n)))


def _auc(values: np.ndarray, is_up: np.ndarray) -> float:
    """Single-feature AUC: P(feature higher on an up move than on a down move).

    Rank-based Mann-Whitney estimator (ties count as 0.5), so it is monotone-invariant and
    needs no model. 0.5 == no separation; far from 0.5 == strong single-feature signal.
    """
    pos = values[is_up]
    neg = values[~is_up]
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
    sum_pos = ranks[is_up].sum()
    auc = (sum_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    return float(auc)


def _rank_for_direction(
    evidence: pd.DataFrame, features: list[str], target: str, min_coverage: float
) -> pd.DataFrame:
    """Rank numeric features by how strongly they separate ``target`` moves from the others."""
    is_target = (evidence["direction"] == target).to_numpy()
    n_total = len(evidence)
    rows: list[dict] = []
    for feat in features:
        col = pd.to_numeric(evidence[feat], errors="coerce").to_numpy(dtype=float)
        valid = ~np.isnan(col)
        coverage = float(valid.mean()) if n_total else 0.0
        if coverage < min_coverage:
            continue
        v = col[valid]
        up_mask = is_target[valid]
        if up_mask.sum() == 0 or (~up_mask).sum() == 0:
            continue
        mean_at_move = float(v[up_mask].mean())
        mean_baseline = float(v[~up_mask].mean())
        pb = _point_biserial(v, up_mask)
        auc = _auc(v, up_mask)
        # Lift: target-class mean over baseline-class mean (robust to sign via abs ratio).
        denom = abs(mean_baseline)
        lift = float(mean_at_move / mean_baseline) if mean_baseline != 0 else float("nan")
        rows.append({
            "feature": feat,
            "coverage": round(coverage, 4),
            "mean_at_move": mean_at_move,
            "mean_baseline": mean_baseline,
            "point_biserial_corr": round(pb, 4) if pb == pb else pb,
            "single_feature_auc": round(auc, 4) if auc == auc else auc,
            "lift": round(lift, 4) if lift == lift else lift,
            "_strength": abs((auc - 0.5)) if auc == auc else -1.0,
            "_n_valid": int(valid.sum()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "feature", "coverage", "mean_at_move", "mean_baseline",
            "point_biserial_corr", "single_feature_auc", "lift"])
    out = out.sort_values("_strength", ascending=False).reset_index(drop=True)
    return out.drop(columns=["_strength", "_n_valid"])


def rank_conditions(evidence: pd.DataFrame, min_coverage: float = 0.05) -> dict[str, pd.DataFrame]:
    """Rank, side by side, which PIT conditions matched the most for up vs down moves.

    Returns ``{"up": DataFrame, "down": DataFrame}``. Each frame has one row per numeric feature
    with columns ``[feature, coverage, mean_at_move, mean_baseline, point_biserial_corr,
    single_feature_auc, lift]``, sorted by ``|single_feature_auc - 0.5|`` descending — strongest
    single-feature separators first. ``mean_at_move`` is the mean among that direction's moves;
    ``mean_baseline`` is the mean among the opposite direction (the comparison class).
    """
    empty = pd.DataFrame(columns=[
        "feature", "coverage", "mean_at_move", "mean_baseline",
        "point_biserial_corr", "single_feature_auc", "lift"])
    if evidence.empty or "direction" not in evidence.columns:
        return {"up": empty.copy(), "down": empty.copy()}
    features = _feature_columns(evidence)
    return {
        "up": _rank_for_direction(evidence, features, "up", min_coverage),
        "down": _rank_for_direction(evidence, features, "down", min_coverage),
    }


def catalyst_match_ranking(evidence: pd.DataFrame) -> pd.DataFrame:
    """Across moves, how often each catalyst type was present for up vs down + direction hit-rate.

    Columns: ``catalyst``, ``n_up`` / ``n_down`` (moves of each direction where it was present),
    ``freq_up`` / ``freq_down`` (fraction of up- resp. down-moves carrying it), ``n_present``,
    ``dir_hit_rate`` (of moves where the catalyst is present, fraction that are up — i.e. how
    often "this catalyst present" co-occurs with an up move). Sorted by total presence.
    """
    cols = ["catalyst", "n_present", "n_up", "n_down", "freq_up", "freq_down", "dir_hit_rate"]
    if evidence.empty or "direction" not in evidence.columns:
        return pd.DataFrame(columns=cols)
    cat_cols = [c for c in evidence.columns if c.startswith(_CAT_PREFIX)]
    if not cat_cols:
        return pd.DataFrame(columns=cols)

    is_up = (evidence["direction"] == "up").to_numpy()
    n_up_total = int(is_up.sum())
    n_down_total = int((~is_up).sum())
    rows: list[dict] = []
    for c in cat_cols:
        present = (pd.to_numeric(evidence[c], errors="coerce").fillna(0) > 0).to_numpy()
        n_up = int((present & is_up).sum())
        n_down = int((present & ~is_up).sum())
        n_present = n_up + n_down
        rows.append({
            "catalyst": c[len(_CAT_PREFIX):],
            "n_present": n_present,
            "n_up": n_up,
            "n_down": n_down,
            "freq_up": round(n_up / n_up_total, 4) if n_up_total else 0.0,
            "freq_down": round(n_down / n_down_total, 4) if n_down_total else 0.0,
            "dir_hit_rate": round(n_up / n_present, 4) if n_present else float("nan"),
        })
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("n_present", ascending=False).reset_index(drop=True)


def evidence_report(
    tickers: Sequence[str] | Iterable[str],
    start: str,
    end: str,
    *,
    client: FMPClient,
    cache: ParquetCache | None = None,
    universe: Universe | None = None,
    settings: Settings | None = None,
    ar_threshold: float = 2.5,
    lookback_days: int = 2,
    min_coverage: float = 0.05,
) -> dict:
    """Bundle: collect move evidence, rank up/down conditions, rank catalyst matches.

    Returns ``{"evidence": DataFrame, "ranking": {"up": ..., "down": ...},
    "catalysts": DataFrame, "n_moves": int, "n_up": int, "n_down": int}``.
    """
    evidence = collect_move_evidence(
        tickers, start, end, client=client, cache=cache, universe=universe, settings=settings,
        ar_threshold=ar_threshold, lookback_days=lookback_days,
    )
    ranking = rank_conditions(evidence, min_coverage=min_coverage)
    catalysts = catalyst_match_ranking(evidence)
    n_up = int((evidence["direction"] == "up").sum()) if not evidence.empty else 0
    n_down = int((evidence["direction"] == "down").sum()) if not evidence.empty else 0
    return {
        "evidence": evidence,
        "ranking": ranking,
        "catalysts": catalysts,
        "n_moves": len(evidence),
        "n_up": n_up,
        "n_down": n_down,
    }
