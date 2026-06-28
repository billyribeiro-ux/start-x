"""Regime-conditional SHORT overlay — the 'both ways' answer: shorts are a BEAR-regime tool, not alpha.

The campaign proved there is no short edge BLENDED over a bull-dominated sample. But a both-ways ML
system never runs shorts blended — it runs them GATED by a regime classifier, on only when the tape is
confirmed bearish. This quantifies the thing that actually matters for that system: short P&L SPLIT BY
REGIME. The regime is the same 200-SMA ±band state as the position book (System #3) — i.e. the switch
the ML must learn to throw.

Two short styles, each measured bull-days vs bear-days:
  * TREND short on the index — short SPY on confirmed-bear days (close < 200-SMA·(1-band)), ride the
    decline; flat in bull. This is a directional bear hedge.
  * the campaign's mean-reversion shorts are the WRONG style for a bear (they fade rallies and get
    squeezed); the right bear short is TREND-following (short weakness that KEEPS falling).

The honest expectation: shorts LOSE on bull days (the drift) and WIN on bear days; blended ≈ 0 (which
is exactly why they fail unconditionally and are valuable only as a regime-activated overlay). Sample
is bear-limited (cache ~2006+, so 2008 tail / 2020 / 2022) — flagged, not hidden.

    python scripts/short_regime_overlay.py [--band 0.0] [--sym SPY]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PRICE_DIR = "data/cache/prices"
TRADING = 252
COST_RT = 0.0002


def _load(sym):
    d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def _stats(r):
    r = r.dropna()
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    sd = r.std(ddof=1)
    mdd = float((eq / eq.cummax() - 1).min())
    yrs = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else float("nan")
    return dict(sharpe=float(r.mean() / sd * np.sqrt(TRADING)) if sd > 0 else float("nan"),
                ann=float(r.mean() * TRADING), total=float(eq.iloc[-1] - 1), maxdd=mdd, cagr=cagr,
                days=len(r), win=float((r > 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sym", default="SPY")
    ap.add_argument("--band", type=float, default=0.0, help="200-SMA hysteresis band (0 = bare cross)")
    args = ap.parse_args()

    d = _load(args.sym)
    c = d.set_index("date")["close"]
    ret = c.pct_change()
    sma = c.rolling(200).mean()
    # confirmed-bear regime (shifted = actionable, no lookahead)
    bear = (c < sma * (1 - args.band)).shift(1).fillna(False).astype(bool)
    # trend short: short the index on bear days, flat on bull days; one cost on each regime entry
    entry = bear & ~bear.shift(1).fillna(False)
    short = (-ret) * bear.astype(float) - entry.astype(float) * COST_RT

    print(f"REGIME-CONDITIONAL SHORT OVERLAY — {args.sym}  {c.index.min().date()}..{c.index.max().date()}")
    print("(bear = close < 200-SMA; short the index on bear days, flat in bull)\n")

    print("=== short stream SPLIT BY REGIME (this is the 'both ways' picture) ===")
    sret = -ret
    on_bear = sret[bear].dropna()
    on_bull = sret[~bear].dropna()
    print(f"  shorting on BEAR days : ann {on_bear.mean()*TRADING*100:+6.1f}%  win {(on_bear>0).mean()*100:4.0f}%  ({len(on_bear)} days, {len(on_bear)/TRADING:.1f}y)")
    print(f"  shorting on BULL days : ann {on_bull.mean()*TRADING*100:+6.1f}%  win {(on_bull>0).mean()*100:4.0f}%  ({len(on_bull)} days)  <- the drag the regime gate removes")
    print(f"  unconditional (always short): ann {sret.mean()*TRADING*100:+.1f}%  <- loses (the bull drift)\n")

    s = _stats(short)
    print("=== the GATED overlay (short only in confirmed bear) ===")
    print(f"  Sharpe {s['sharpe']:+.2f}  ann {s['ann']*100:+.1f}%  total {s['total']*100:+.0f}%  "
          f"maxDD {s['maxdd']*100:.0f}%  exposure {bear.mean()*100:.0f}% of days\n")

    # --- the REAL construction: a trailing TREND short that rides sustained declines, exits rallies ---
    o = d.set_index("date")["open"]; hi = d.set_index("date")["high"]; lo = d.set_index("date")["low"]
    pc = c.shift(1)
    tr = pd.concat([(hi - lo), (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    cl = c.to_numpy(); op = o.to_numpy(); hh = hi.to_numpy(); ll = lo.to_numpy()
    av = atr.to_numpy(); be = bear.to_numpy(); lo50 = c.rolling(50).min().to_numpy()
    idx = c.index; n = len(cl)
    trend = np.zeros(n)
    i = 0
    while i < n - 1:
        # enter short: confirmed bear AND a fresh 50-day low (downside follow-through), ride chandelier
        if be[i] and np.isfinite(av[i]) and av[i] > 0 and cl[i] <= lo50[i]:
            entry = cl[i]; trough = entry; j = i + 1
            trend[i] -= COST_RT
            while j < n:
                trough = min(trough, ll[j])
                cover_lvl = trough + 3.0 * av[i]              # 3-ATR trailing cover
                if hh[j] >= cover_lvl or not be[j]:           # rally through trail OR regime flips bull
                    fill = max(cover_lvl, op[j]) if hh[j] >= cover_lvl else cl[j]
                    trend[j] += (cl[j - 1] - fill) / entry    # final-day P&L piece
                    break
                trend[j] += (cl[j - 1] - cl[j]) / entry        # daily short P&L while riding
                j += 1
            i = j + 1
        else:
            i += 1
    trend = pd.Series(trend, index=idx)
    ts = _stats(trend)
    print("=== TRAILING TREND short (short fresh lows in a bear, ride a 3-ATR chandelier) ===")
    print(f"  Sharpe {ts.get('sharpe', float('nan')):+.2f}  ann {ts.get('ann', 0)*100:+.1f}%  "
          f"total {ts.get('total', 0)*100:+.0f}%  maxDD {ts.get('maxdd', 0)*100:.0f}%\n")

    print("=== by YEAR: naive daily-hold short  vs  trailing trend short ===")
    yrs = pd.DatetimeIndex(short.index).year
    tyr = pd.DatetimeIndex(trend.index).year
    for y in sorted(set(yrs)):
        bd = int(bear[pd.DatetimeIndex(bear.index).year == y].sum())
        if bd < 5:
            continue
        naive = (1 + short[yrs == y]).prod() - 1
        tnd = (1 + trend[tyr == y]).prod() - 1
        flag = "  <- sustained bear" if tnd > 0.05 else ""
        print(f"  {y}: bear-days {bd:3d}   naive {naive*100:+6.1f}%   trend {tnd*100:+6.1f}%{flag}")
    print("\nTAKEAWAY for the both-ways ML: the short is a REGIME-ACTIVATED overlay. Its whole job is to")
    print("be ON in the bear and OFF in the bull — so the ML's real task is REGIME DETECTION (throw the")
    print("switch early), not finding a standalone short signal that works in every tape (none does).")


if __name__ == "__main__":
    main()
