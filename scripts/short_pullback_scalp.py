"""SHORT the over-extension, SCALP the retrace to the mean — the user's mean-reversion thesis, tested fairly.

"Nothing goes up straight without a pullback/retrace to the mean." The prior overbought study measured
the price N days LATER (the endpoint) and so held right through the retrace into the resumed uptrend —
an unfair test of a SCALP. This tests the trade as it is actually run: short an over-extension, COVER
FAST on a quick pullback (profit target a fraction of an ATR below entry), a tight stop above, a 1-3 day
cap. The exact mirror of how the long IBS edge scalps the bounce. Path-aware, gap-aware fills.

A real edge must clear the firewall OOS net of cost+borrow, not just win in-sample. TRAIN 2018-22 picks
the config; TEST 2023->2026-06-25 is untouched. n_trials counts every (trigger x target x stop x cap).

    python scripts/short_pullback_scalp.py [--max-names N]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import itertools
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from startx.strategy.mean_reversion import atr, rsi

PRICE_DIR = "data/cache/prices"
TRAIN_END = pd.Timestamp("2022-12-31")
TEST_START = pd.Timestamp("2023-01-01")
COST_RT = 0.0004              # 4bp round-trip slippage+commission (realistic single-name)
BORROW_ANN = 0.05            # 5%/yr hard-to-borrow stress
TRADING = 252


def _names(max_names):
    fs = sorted(glob.glob(f"{PRICE_DIR}/*.parquet"))
    syms = [os.path.basename(f)[:-8] for f in fs]
    syms = [s for s in syms if not s.startswith("_") and s not in ("SPY", "GLD", "UUP")]
    return syms[:max_names] if max_names else syms


def _load(sym):
    try:
        d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    except Exception:
        return None
    if len(d) < 300:
        return None
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def _triggers(d):
    """Over-extension triggers — 'it went up too straight'. Returns {name: bool mask}."""
    c, h, l, o = d["close"], d["high"], d["low"], d["open"]
    ibs = ((c - l) / (h - l).replace(0, np.nan)).fillna(0.5)
    r2 = rsi(c, 2)
    sma20 = c.rolling(20).mean()
    up = (c > c.shift(1))
    up3 = up & up.shift(1) & up.shift(2) & (c >= 1.05 * c.shift(3))     # 3 up days + >=5% surge
    return {
        "rsi2_95":  (r2 >= 95) & (c > sma20),
        "ibs_surge": (ibs >= 0.9) & up & (c >= 1.03 * c.shift(2)),
        "up3_surge": up3,
        "ext10":    c >= 1.10 * sma20,
    }


def _book(trades, dates):
    """Per-trade returns -> mean expectancy + a by-trade annualized Sharpe (independent-bet proxy)."""
    if len(trades) < 30:
        return None
    r = np.array([t[1] for t in trades])
    bars = np.array([t[0] for t in trades]).mean()
    sd = r.std(ddof=1)
    # bets/yr ~ 252/avg_hold; Sharpe of the per-trade stream scaled to annual
    sh = r.mean() / sd * np.sqrt(TRADING / max(bars, 1)) if sd > 0 else float("nan")
    return dict(n=len(r), exp=r.mean(), win=(r > 0).mean(), sh=sh, avg_bars=bars, tot=r.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-names", type=int, default=None)
    args = ap.parse_args()
    names = _names(args.max_names)
    borrow_d = BORROW_ANN / TRADING

    TARGETS = [0.5, 1.0]; STOPS = [1.0, 1.5]; CAPS = [2, 3]
    trig_names = ["rsi2_95", "ibs_surge", "up3_surge", "ext10"]
    configs = list(itertools.product(trig_names, TARGETS, STOPS, CAPS))
    n_trials = len(configs)

    # collect trades per config, split by entry-date window (one pass, entry dates tracked explicitly)
    tr_trades = {cfg: [] for cfg in configs}
    te_trades = {cfg: [] for cfg in configs}
    for sym in names:
        d = _load(sym)
        if d is None:
            continue
        trg = _triggers(d)
        c = d["close"].to_numpy(float); h = d["high"].to_numpy(float)
        l = d["low"].to_numpy(float); o = d["open"].to_numpy(float)
        a = atr(d, 14).to_numpy(float); dts = d["date"].to_numpy()
        n = len(c)
        for tname in trig_names:
            base = trg[tname].fillna(False).to_numpy()
            idx0 = np.flatnonzero(base); idx0 = idx0[idx0 >= 200]
            if len(idx0) == 0:
                continue
            for tm, sm, cap in itertools.product(TARGETS, STOPS, CAPS):
                cfg = (tname, tm, sm, cap)
                last_exit = -1
                for i in idx0:
                    if i <= last_exit or not np.isfinite(a[i]) or a[i] <= 0 or i + 1 >= n:
                        continue
                    e = c[i]; av = a[i]; target = e - tm * av; stop = e + sm * av
                    cover = None; last = min(i + cap, n - 1)
                    for j in range(i + 1, last + 1):
                        if h[j] >= stop:
                            cover = max(stop, o[j]); jx = j; break
                        if l[j] <= target:
                            cover = min(target, o[j]); jx = j; break
                    else:
                        cover = c[last]; jx = last
                    bars = jx - i
                    ret = (e - cover) / e - COST_RT - borrow_d * bars
                    (tr_trades if dts[i] <= TRAIN_END.to_datetime64() else te_trades)[cfg].append((bars, ret))
                    last_exit = jx

    print(f"SHORT PULLBACK-SCALP — {len(names)} names, {n_trials} configs "
          f"(cost {COST_RT*1e4:.0f}bp RT + borrow {BORROW_ANN:.0%}/yr)\n")
    # rank by TRAIN expectancy, then show TEST (OOS) for the winner + the firewall
    ranked = []
    for cfg in configs:
        bt = _book(tr_trades[cfg], None)
        if bt:
            ranked.append((cfg, bt))
    ranked.sort(key=lambda x: -x[1]["exp"])
    print("=== TRAIN (2018-22) — top configs by per-trade expectancy (net) ===")
    print(f"{'trigger/tgt/stop/cap':28}{'n':>7}{'exp%':>8}{'win%':>7}{'Sharpe':>8}{'bars':>6}")
    for cfg, b in ranked[:8]:
        lab = f"{cfg[0]}/{cfg[1]}/{cfg[2]}/{cfg[3]}"
        print(f"{lab:28}{b['n']:>7}{b['exp']*100:>8.3f}{b['win']*100:>7.1f}{b['sh']:>8.2f}{b['avg_bars']:>6.1f}")

    best = ranked[0][0]
    print(f"\n>>> SELECTED on TRAIN: {best}")
    from startx.validation.metrics import deflated_sharpe
    from qedge.validation import survival_gate
    for lab, bucket in [("TRAIN", tr_trades), ("TEST (OOS)", te_trades)]:
        b = _book(bucket[best], None)
        if not b:
            print(f"  {lab}: <30 trades"); continue
        print(f"  {lab:10}: n={b['n']:6d}  exp={b['exp']*100:+.3f}%  win={b['win']*100:.1f}%  "
              f"Sharpe={b['sh']:+.2f}  avg_bars={b['avg_bars']:.1f}  total={b['tot']*100:+.0f}%")
    # firewall on the OOS per-trade stream
    te = te_trades[best]
    if len(te) > 30:
        r = np.array([t[1] for t in te])
        dsr = deflated_sharpe(r, n_trials=n_trials)
        g = survival_gate(r, n_trials=n_trials)
        print("\n================ FIREWALL (OOS per-trade) ================")
        print(f"  DSR={dsr:.3f}  gate.DSR={g.deflated_sharpe:.3f}  PBO={g.pbo:.3f}  passed={g.passed}  ({g.verdict})")
        print(f"  VERDICT: {'REAL — short pullback edge survives OOS!' if g.passed else 'still fails OOS net of cost+borrow.'}")


if __name__ == "__main__":
    main()
