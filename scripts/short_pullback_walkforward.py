"""PRE-REGISTERED walk-forward of the weak-tape pullback-scalp short — confirm or KILL it. (KILLED.)

The rule was LOCKED and tested on data the selection never touched. The pre-registered run first looked
spectacular (calendar-book Sharpe ~1.9, DSR 1.0) — which was TOO GOOD, and adversarial verification
proved it a MIRAGE:

    SHORT a name when close >= 1.10*SMA20 (>=10% over-extended), ONLY while SPY < its 20-day SMA;
    cover on a 1-ATR retrace / 1-ATR stop, 3-day cap. Net of 4bp + 5%/yr borrow.

  * On the FULL universe the per-trade returns run from -267% to +98% — microcap / bad-data / delisted-
    to-zero JUNK you can never actually short. Median trade is +0.01% (zero). The high Sharpe is those
    outliers + a BREADTH ILLUSION (averaging thousands of correlated names crushes the daily vol).
  * It is NOT even beta: corr to short-SPY ~+0.01, short-SPY benchmark Sharpe -0.65 on the same days.
  * On LIQUID, tradeable names only (>$5, >$20M/day ADV): expectancy -0.17%/trade, book Sharpe -0.01.
    The edge VANISHES. => no tradeable short edge. The firewall (and a too-good-to-be-true alarm) worked.

This script reports FULL vs LIQUID + the short-SPY beta benchmark so the kill is reproducible. Default
is liquid-only (the honest, tradeable verdict); pass --include-junk to see the mirage.

    python scripts/short_pullback_walkforward.py [--max-names N] [--borrow 0.05] [--include-junk]
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

from startx.strategy.mean_reversion import atr
from startx.validation.metrics import deflated_sharpe

PRICE_DIR = "data/cache/prices"
COST_RT = 0.0004
TRADING = 252


def _load(sym):
    d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def _weak_tape():
    spc = _load("SPY").set_index("date")["close"]
    return (spc < spc.rolling(20).mean()).shift(1).fillna(False)   # known at entry, no lookahead


def _collect(names, weak, borrow_d, liquid_only=True):
    """LOCKED rule -> trades (entry_date, exit_date, ret, bars), gated to weak tape.

    liquid_only filters to ACTUALLY TRADEABLE names (median close >$5 and median dollar-volume
    >$20M/day) — without it, microcap / bad-data / delisted-to-zero junk manufactures a fake edge.
    """
    wt = weak.to_dict()
    rows = []
    for sym in names:
        try:
            d = _load(sym)
        except Exception:
            continue
        if len(d) < 300:
            continue
        cc = d["close"].to_numpy(float); vv = d["volume"].to_numpy(float)
        if liquid_only and (np.nanmedian(cc) < 5 or np.nanmedian(cc * vv) < 2e7):
            continue
        c = d["close"].to_numpy(float); h = d["high"].to_numpy(float)
        l = d["low"].to_numpy(float); o = d["open"].to_numpy(float)
        a = atr(d, 14).to_numpy(float); dts = d["date"].to_numpy()
        dd = pd.to_datetime(d["date"])
        sma20 = d["close"].rolling(20).mean().to_numpy()
        sig = np.flatnonzero(c >= 1.10 * sma20); sig = sig[sig >= 200]
        last = -1; n = len(c)
        for i in sig:
            if i <= last or not np.isfinite(a[i]) or a[i] <= 0 or i + 1 >= n:
                continue
            if not wt.get(dd.iloc[i], False):            # GATE: weak tape only
                continue
            e = c[i]; av = a[i]; tgt = e - av; stp = e + av
            cover = None; lb = min(i + 3, n - 1)
            for j in range(i + 1, lb + 1):
                if h[j] >= stp:
                    cover = max(stp, o[j]); jx = j; break
                if l[j] <= tgt:
                    cover = min(tgt, o[j]); jx = j; break
            else:
                cover = c[lb]; jx = lb
            bars = jx - i
            ret = (e - cover) / e - COST_RT - borrow_d * bars
            rows.append((dts[i], dts[jx], ret, bars))
            last = jx
    return pd.DataFrame(rows, columns=["entry", "exit", "ret", "bars"])


def _daily_book(tr):
    """Calendar-time equal-weight daily book: each trade spreads ret/bars over (entry, exit]."""
    if tr.empty:
        return pd.Series(dtype=float)
    from collections import defaultdict
    s = defaultdict(float); cnt = defaultdict(int)
    for e, x, r, b in zip(tr["entry"], tr["exit"], tr["ret"], tr["bars"]):
        b = max(int(b), 1); per = r / b
        for day in pd.bdate_range(pd.Timestamp(e) + pd.Timedelta(days=1), pd.Timestamp(x)):
            s[day] += per; cnt[day] += 1
    idx = sorted(s)
    return pd.Series([s[d] / cnt[d] for d in idx], index=pd.DatetimeIndex(idx))


def _sh(r):
    r = r.dropna()
    if len(r) < 20:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING)) if sd > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-names", type=int, default=None)
    ap.add_argument("--borrow", type=float, default=0.05)
    ap.add_argument("--include-junk", action="store_true", help="don't filter for liquidity (shows the mirage)")
    args = ap.parse_args()
    weak = _weak_tape()
    names = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{PRICE_DIR}/*.parquet"))]
    names = [s for s in names if not s.startswith("_") and s not in ("SPY", "GLD", "UUP")]
    if args.max_names:
        names = names[:args.max_names]

    # short-SPY daily return (the beta-timing benchmark on the same active days)
    spc = _load("SPY").set_index("date")["close"]; sspy = (-spc.pct_change())

    print("PRE-REGISTERED WALK-FORWARD — weak-tape pullback-scalp (LOCKED rule, n_trials=1)\n")
    for liquid in ([True] if not args.include_junk else [False, True]):
        tr = _collect(names, weak, args.borrow / TRADING, liquid_only=liquid)
        book = _daily_book(tr)
        tag = "LIQUID only (>$5, >$20M ADV — tradeable)" if liquid else "FULL universe (incl. microcap/junk)"
        if len(book) < 30:
            print(f"=== {tag}: <30 book-days ==="); continue
        df = pd.concat([book.rename("b"), sspy.reindex(book.index).rename("s")], axis=1).dropna()
        beta = float(np.polyfit(df["s"], df["b"], 1)[0]) if len(df) > 20 else float("nan")
        resid = df["b"] - beta * df["s"]
        print(f"=== {tag} ===")
        print(f"  {len(tr)} trades  median-ret {tr.ret.median()*100:+.3f}%  worst {tr.ret.min()*100:+.0f}%  best {tr.ret.max()*100:+.0f}%")
        print(f"  per-trade exp {tr.ret.mean()*100:+.3f}%  |  daily-book Sharpe {_sh(book):+.2f}  ann {book.mean()*TRADING*100:+.1f}%")
        print(f"  vs short-SPY benchmark Sharpe {_sh(df['s']):+.2f} | corr {df['b'].corr(df['s']):+.2f} | beta-neutral resid Sharpe {_sh(resid):+.2f}")
        print(f"  DSR(n_trials=1) {deflated_sharpe(book.values, n_trials=1):.3f}\n")

    print("VERDICT: on LIQUID tradeable names the edge is NEGATIVE (Sharpe ~0) — the earlier 'positive'")
    print("was microcap/bad-data outliers (-267%..+98% trades) + a breadth illusion. NO tradeable short edge.")


if __name__ == "__main__":
    main()
