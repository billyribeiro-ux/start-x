"""SHORT the OVERBOUGHT — the symmetric mirror of the long IBS edge (the last untried short angle).

Every prior short angle shorted WEAKNESS and died because weak names BOUNCE (oversold mean-reversion
— the same force that powers our long IBS dip-buy). This asks the mirror: do OVERBOUGHT / over-extended
/ blow-off single names revert DOWN, so that SHORTING STRENGTH is the real edge?

A short wins when the forward return is NEGATIVE. So the decisive first cut is the forward-return
SIGNATURE after each overbought trigger: if over-extension predicts negative forward drift, there's an
edge to firewall; if it predicts positive/flat drift (overbought keeps running — momentum), the mirror
is dead on arrival. Conditioned on trend (the long edge gates on UPTREND; the mirror short might gate on
DOWNTREND). Net of ~2bp + borrow. TRAIN 2018-22 / TEST 2023->2026-06-25. Honest n_trials.

    python scripts/short_overbought_study.py [--max-names N] [--firewall]
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

from startx.strategy.mean_reversion import rsi

PRICE_DIR = "data/cache/prices"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
BORROW_ANN = 0.05            # hard-to-borrow stress on the short
HOLD = 5                     # short-swing hold for the book


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


# Overbought / over-extension triggers (mirror of the long oversold setups). Each returns a bool mask.
def _signals(d):
    c, h, l, o, v = (d["close"], d["high"], d["low"], d["open"], d["volume"])
    ibs = ((c - l) / (h - l).replace(0, np.nan)).fillna(0.5)
    r2 = rsi(c, 2)
    sma20 = c.rolling(20).mean(); sma50 = c.rolling(50).mean(); sma200 = c.rolling(200).mean()
    sd20 = c.rolling(20).std()
    volr = v / v.rolling(50).mean()
    upgap = o / c.shift(1) - 1.0
    return {
        "ibs_high":      ibs >= 0.90,
        "ibs_x":         ibs >= 0.95,
        "rsi2_ob":       r2 >= 95,
        "rsi2_x":        r2 >= 98,
        "ext20_15":      c >= 1.15 * sma20,
        "ext50_25":      c >= 1.25 * sma50,
        "bb_upper":      c > sma20 + 2.5 * sd20,
        "blowoff_gap":   (upgap >= 0.07) & (volr >= 2.0),
    }, sma200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-names", type=int, default=None)
    ap.add_argument("--firewall", action="store_true", help="firewall any setup with negative drift")
    args = ap.parse_args()
    names = _names(args.max_names)

    rows = []
    for sym in names:
        d = _load(sym)
        if d is None:
            continue
        sigs, sma200 = _signals(d)
        c = d["close"].to_numpy(float)
        dt = d["date"].to_numpy()
        up = (d["close"] > sma200).to_numpy()          # uptrend regime
        n = len(c)
        for name, m in sigs.items():
            idx = np.flatnonzero(m.fillna(False).to_numpy())
            for i in idx:
                if i + 10 >= n or i < 200:
                    continue
                rows.append((name, dt[i], bool(up[i]),
                             c[i + 1] / c[i] - 1, c[i + 3] / c[i] - 1,
                             c[i + 5] / c[i] - 1, c[i + 10] / c[i] - 1))
    E = pd.DataFrame(rows, columns=["setup", "date", "uptrend", "f1", "f3", "f5", "f10"])
    E["date"] = pd.to_datetime(E["date"])
    print(f"SHORT-THE-OVERBOUGHT — {len(names)} names, {len(E)} signals  (a SHORT wins if fwd ret < 0)\n")

    # --- forward-return SIGNATURE (the decisive cut): mean forward return after each trigger -------
    borrow_d = BORROW_ANN / 252
    print("=== forward UNDERLYING return after trigger (NEGATIVE = short wins). split by trend, TRAIN|TEST ===")
    print(f"{'setup':12}{'trend':8}{'n_tr':>7}{'f5_tr%':>9}{'f10_tr%':>9}{'n_te':>7}{'f5_te%':>9}{'f10_te%':>9}")
    promising = []
    for setup in E["setup"].unique():
        for trend, sub0 in [("ALL", E[E.setup == setup]),
                            ("up", E[(E.setup == setup) & E.uptrend]),
                            ("down", E[(E.setup == setup) & ~E.uptrend])]:
            tr = sub0[sub0.date <= TRAIN_END]; te = sub0[sub0.date >= TEST_START]
            if len(tr) < 50 or len(te) < 20:
                continue
            print(f"{setup:12}{trend:8}{len(tr):>7}{tr.f5.mean()*100:>9.2f}{tr.f10.mean()*100:>9.2f}"
                  f"{len(te):>7}{te.f5.mean()*100:>9.2f}{te.f10.mean()*100:>9.2f}")
            # short edge needs forward return negative enough to beat cost+borrow in BOTH windows
            need = 0.0002 + borrow_d * HOLD                       # ~2bp + 5d borrow
            if tr.f5.mean() < -need and te.f5.mean() < -need:
                promising.append((setup, trend))
        print()

    if not promising:
        print(">>> NO overbought setup shows negative forward drift beating cost+borrow in BOTH windows.")
        print(">>> Overbought single names KEEP RISING (momentum), they do not revert down. Mirror DEAD.")
        return
    print(f">>> PROMISING (negative drift beats cost in train+test): {promising}")
    if args.firewall:
        from startx.validation.metrics import deflated_sharpe
        from qedge.validation import survival_gate
        n_trials = E["setup"].nunique() * 3
        for setup, trend in promising:
            sub = E[E.setup == setup]
            sub = sub[sub.uptrend] if trend == "up" else (sub[~sub.uptrend] if trend == "down" else sub)
            te = sub[sub.date >= TEST_START]
            short_ret = -(te["f5"].to_numpy()) - borrow_d * HOLD - 0.0002       # short P&L net
            book = pd.Series(short_ret, index=pd.to_datetime(te["date"].to_numpy())).groupby(level=0).mean()
            sh = book.mean() / book.std(ddof=1) * np.sqrt(252 / HOLD) if book.std() > 0 else float("nan")
            dsr = deflated_sharpe(book.values, n_trials=n_trials) if len(book) > 30 else float("nan")
            g = survival_gate(book.values, n_trials=n_trials)
            print(f"  [{setup}/{trend}] OOS book Sharpe {sh:+.2f}  DSR {dsr:.3f}  "
                  f"gate.DSR {g.deflated_sharpe:.3f} PBO {g.pbo:.3f} passed={g.passed}")


if __name__ == "__main__":
    main()
