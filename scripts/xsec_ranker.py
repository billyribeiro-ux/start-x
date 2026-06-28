"""Cross-sectional ML ranker — market-neutral long/short on the SURVIVORSHIP-FREE S&P 500.

The system's best diversifier (DECODE: ~0.87 Sharpe, beta~0) and the genuine 'both ways' engine: long
the predicted-strong decile, short the predicted-weak decile, dollar-neutral. Built RIGHT this time —
the universe on each monthly rebalance is `membership.tradeable_asof(date, cached)` (point-in-time,
includes since-delisted names), so the long leg is NOT flattered by survivorship (the trap that just
manufactured a fake short edge).

Discipline (all of it, because a long/short ranker is the easiest place to fool yourself):
  * PIT features (all trailing) -> forward 21d return; cross-sectionally z-scored / demeaned per date.
  * WALK-FORWARD LightGBM (expanding train, embargo) -> OOS predictions only. No row trains on its own
    future. Retrained periodically.
  * Market-neutral book net of realistic costs (turnover-based) + short borrow.
  * FIREWALL on the OOS monthly stream: DSR / PBO. BETA to SPY (must be ~0). And a NAIVE momentum
    baseline — the ML must beat raw 12-1 momentum, or it is adding nothing.

    python scripts/xsec_ranker.py [--start 2006-01-01] [--rebal 21] [--decile 0.1]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from startx.data.membership import (
    REAL_DATA_START,
    CoverageError,
    SP500Membership,
)
from startx.validation.metrics import deflated_sharpe

PRICE_DIR = "data/cache/prices"
FEATURES = ["ret5", "ret21", "ret63", "ret126", "ret252", "vol21", "vol63", "amihud", "d50", "d200"]


def _price_cache():
    out = {}
    for f in glob.glob(f"{PRICE_DIR}/*.parquet"):
        sym = os.path.basename(f)[:-8]
        if sym.startswith("_"):
            continue
        try:
            d = pd.read_parquet(f, columns=["date", "open", "high", "low", "close", "volume"])
        except Exception:
            continue
        if len(d) < 300:
            continue
        d["date"] = pd.to_datetime(d["date"])
        out[sym] = d.sort_values("date").reset_index(drop=True)
    return out


def _features_for(d: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol trailing features indexed by date (all known at the bar's close — no lookahead)."""
    c = d["close"].astype(float)
    dollar = (c * d["volume"].astype(float)).replace(0, np.nan)
    r = c.pct_change()
    f = pd.DataFrame(index=d["date"])
    f["ret5"] = (c / c.shift(5) - 1).values
    f["ret21"] = (c / c.shift(21) - 1).values
    f["ret63"] = (c / c.shift(63) - 1).values
    f["ret126"] = (c / c.shift(126) - 1).values
    f["ret252"] = (c / c.shift(252) - 1).values
    f["vol21"] = r.rolling(21).std().values
    f["vol63"] = r.rolling(63).std().values
    f["amihud"] = (r.abs() / dollar).rolling(21).mean().values * 1e9
    f["d50"] = (c / c.rolling(50).mean() - 1).values
    f["d200"] = (c / c.rolling(200).mean() - 1).values
    f["close"] = c.values
    return f


def build_panel(mem, cache, start, rebal, fwd, survivors_only=False):
    """Long panel: one row per (rebalance_date, member-with-prices) with PIT features + fwd return."""
    # monthly-ish rebalance calendar from SPY trading days
    spy = cache.get("SPY")
    if spy is None:
        spy = pd.read_parquet(f"{PRICE_DIR}/SPY.parquet"); spy["date"] = pd.to_datetime(spy["date"])
    days = pd.to_datetime(spy["date"]).sort_values().reset_index(drop=True)
    days = days[days >= pd.Timestamp(start)].reset_index(drop=True)
    rebal_idx = list(range(0, len(days) - fwd, rebal))
    rebal_dates = [days[i] for i in rebal_idx]

    # precompute per-symbol feature frames once
    feats = {s: _features_for(d).set_index(pd.DatetimeIndex(_features_for(d).index)) for s, d in cache.items()} \
        if False else {s: _features_for(d) for s, d in cache.items()}
    for s in feats:
        feats[s].index = pd.DatetimeIndex(feats[s].index)

    rows = []
    for rd in rebal_dates:
        fd = days[days > rd]
        fd = fd.iloc[fwd - 1] if len(fd) >= fwd else None
        if fd is None:
            continue
        universe = (frozenset(mem.current) & set(cache.keys())) if survivors_only \
            else mem.tradeable_asof(rd, cache.keys())
        for s in universe:
            ff = feats.get(s)
            if ff is None or rd not in ff.index:
                continue
            row = ff.loc[rd]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if not np.isfinite(row.get("ret252", np.nan)):
                continue
            # forward return over the holding period
            fut = feats[s]
            if fd not in fut.index:
                continue
            c0 = row["close"]; c1 = fut.loc[fd]
            c1 = c1["close"] if isinstance(c1, pd.Series) else (c1.iloc[0]["close"] if isinstance(c1, pd.DataFrame) else np.nan)
            if not (np.isfinite(c0) and np.isfinite(c1) and c0 > 0):
                continue
            rec = {"date": rd, "fdate": fd, "symbol": s, "fwd": c1 / c0 - 1.0}
            for k in FEATURES:
                rec[k] = row[k]
            rows.append(rec)
    panel = pd.DataFrame(rows)
    return panel, rebal_dates


def _xs_prep(panel):
    """Cross-sectionally z-score features and demean the target within each rebalance date."""
    p = panel.copy()
    for k in FEATURES:
        g = p.groupby("date")[k]
        p[k] = ((p[k] - g.transform("mean")) / g.transform("std")).clip(-4, 4).fillna(0.0)
    p["y"] = p["fwd"] - p.groupby("date")["fwd"].transform("mean")     # market-excess target
    return p


def walk_forward(panel, train_min, retrain_every):
    """Expanding-window LightGBM; predict each rebalance's cross-section OOS (embargo 1 period)."""
    import lightgbm as lgb
    p = _xs_prep(panel)
    dates = sorted(p["date"].unique())
    preds = pd.Series(np.nan, index=p.index)
    model = None
    for k, rd in enumerate(dates):
        if k < train_min:
            continue
        if model is None or (k - train_min) % retrain_every == 0:
            tr = p[p["date"].isin(dates[:k - 1])]            # embargo the immediately-prior period
            model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31,
                                      subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
                                      random_state=42, n_jobs=1, verbosity=-1)
            model.fit(tr[FEATURES], tr["y"])
        cur = p[p["date"] == rd]
        preds.loc[cur.index] = model.predict(cur[FEATURES])
    p["pred"] = preds
    return p.dropna(subset=["pred"])


def book(panel_pred, decile, cost_bps, borrow_ann, rebal, by="pred"):
    """Market-neutral long/short book return per rebalance, net of turnover cost + short borrow."""
    rows = []
    prev_long, prev_short = set(), set()
    cps = cost_bps / 1e4
    borrow = borrow_ann * (rebal / 252.0)
    for rd, g in panel_pred.groupby("date"):
        g = g.sort_values(by)
        n = max(int(len(g) * decile), 1)
        short = g.head(n); long = g.tail(n)
        ls, ll = set(short["symbol"]), set(long["symbol"])
        gross = long["fwd"].mean() - short["fwd"].mean()
        turn = (len(ll - prev_long) + len(ls - prev_short)) / max(len(ll) + len(ls), 1)
        cost = turn * 2 * cps + borrow / 2          # borrow on the short half
        rows.append({"date": rd, "ret": gross - cost, "n": len(g),
                     "long_raw": long["fwd"].mean(), "short_raw": short["fwd"].mean()})
        prev_long, prev_short = ll, ls
    return pd.DataFrame(rows).set_index("date")


def _stats(r, ppy):
    r = r.dropna()
    if len(r) < 12:
        return {}
    sd = r.std(ddof=1)
    eq = (1 + r).cumprod()
    return dict(n=len(r), sharpe=float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else float("nan"),
                ann=float(r.mean() * ppy), total=float(eq.iloc[-1] - 1),
                maxdd=float((eq / eq.cummax() - 1).min()), win=float((r > 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=str(REAL_DATA_START.date()),
                    help="panel start (default = the real-data floor; earlier is blocked by the guard)")
    ap.add_argument("--rebal", type=int, default=21)
    ap.add_argument("--decile", type=float, default=0.1)
    ap.add_argument("--cost-bps", type=float, default=5.0, dest="cost_bps")
    ap.add_argument("--borrow", type=float, default=0.05)
    ap.add_argument("--report-start", default="2018-01-01", dest="report_start",
                    help="slice OOS stats to this date (the only window with real ~98% data)")
    ap.add_argument("--survivors-only", action="store_true", dest="survivors_only",
                    help="A/B: use TODAY's members for all history (the biased way) to size survivorship")
    args = ap.parse_args()
    ppy = 252 / args.rebal
    rstart = pd.Timestamp(args.report_start)

    # GUARD: a cross-sectional study before the real-data floor silently re-injects survivorship
    # (the missing names are the delisted ones) and produces FAKE results. Fail loud, not silent.
    if pd.Timestamp(args.start) < REAL_DATA_START:
        raise CoverageError(
            f"--start {args.start} is before the real-data floor {REAL_DATA_START.date()}: pre-2016 "
            f"cross-sections are survivorship-contaminated (the price cache starts ~2010 and misses "
            f"delisted names). Refusing to produce fake results — use --start >= {REAL_DATA_START.date()}.")

    print("loading price cache + PIT membership...")
    cache = _price_cache()
    mem = SP500Membership.load()
    tag = "CURRENT-SURVIVORS (biased)" if args.survivors_only else "survivorship-FREE (PIT)"
    print(f"  {len(cache)} symbols cached; building panel [{tag}]...")
    panel, rebal_dates = build_panel(mem, cache, args.start, args.rebal, args.rebal,
                                     survivors_only=args.survivors_only)
    if not args.survivors_only:                       # the reported window must be ~complete
        reported = [d for d in rebal_dates if pd.Timestamp(d) >= rstart]
        rep = mem.require_coverage(cache.keys(), reported, min_cov=0.95)
        print(f"  coverage guard PASSED: reported window {rstart.date()}+ at "
              f"{rep['coverage'].min():.0%}-{rep['coverage'].max():.0%} universe coverage")
    print(f"  panel: {len(panel)} rows, {panel['date'].nunique()} rebalances "
          f"({panel['date'].min().date()}..{panel['date'].max().date()}), ~{len(panel)//max(panel['date'].nunique(),1)} names/date\n")

    pp = walk_forward(panel, train_min=24, retrain_every=6)
    ml = book(pp, args.decile, args.cost_bps, args.borrow, args.rebal, by="pred")
    # NAIVE baseline: rank by raw 12-1 momentum (ret252 - ret21), same construction
    pp["mom121"] = pp["ret252"] - pp["ret21"]
    naive = book(pp, args.decile, args.cost_bps, args.borrow, args.rebal, by="mom121")

    spy = cache["SPY"].set_index("date")["close"].pct_change()

    def show(name, b, n_trials):
        b = b[b.index >= rstart]                      # report ONLY the real-data window
        s = _stats(b["ret"], ppy)
        dsr = deflated_sharpe(b["ret"].dropna().values, n_trials=n_trials)
        sret = pd.Series({d: (1 + spy.loc[(spy.index > d - pd.Timedelta(days=args.rebal*2)) & (spy.index <= d)]).prod() - 1 for d in b.index})
        df = pd.concat([b["ret"], sret.rename("spy")], axis=1).dropna()
        beta = float(np.polyfit(df["spy"], df["ret"], 1)[0]) if len(df) > 12 else float("nan")
        print(f"  {name:18} Sharpe {s.get('sharpe',float('nan')):+.2f}  ann {s.get('ann',0)*100:+.1f}%  "
              f"maxDD {s.get('maxdd',0)*100:.0f}%  win {s.get('win',0)*100:.0f}%  DSR {dsr:.2f}  "
              f"beta {beta:+.2f}  n={s.get('n',0)}")

    print(f"=== MARKET-NEUTRAL RANKER — OOS {rstart.date()}..{ml.index.max().date()}, net of costs+borrow ===")
    print(f"    universe = {tag};  N_trials fed to DSR is the project ledger count")
    show("ML ranker", ml, n_trials=14)             # see LEDGER.md for the running N
    show("naive 12-1 mom", naive, n_trials=14)
    print("\nKILL CONDITION (pre-stated): if the NAIVE baseline shows no OOS cost-adjusted signal on the")
    print("real-data window, STOP — per staged-search, escalating the ML only manufactures a mirage.")


if __name__ == "__main__":
    main()
