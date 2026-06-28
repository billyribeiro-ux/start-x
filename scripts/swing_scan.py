"""Swing scanner — P1 entry: tradeability gate over the v1 universe (Charter v1).

Checkpoint (per the charter, "After P1: show the tradeability scores for the v1 universe + 5 rejected
names with reasons"). Runs Layer 0 on the v1 ETF universe plus a sample of single names (liquid members
AND deliberately illiquid / idiosyncratic ones) so the gate's rejections are visible. PIT as-of = the
latest cached bar. Dealer-GEX / options components are SEAMS (no OPRA data) — surfaced, not faked.

    python scripts/swing_scan.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from startx.data.membership import SP500Membership
from startx.scanner.tradeability import MACRO_FACTORS, score_symbol

PRICE_DIR = "data/cache/prices"
V1_ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "SMH", "XLV", "XLU"]
# a sample of single names to show the gate REJECTING (mix of liquid members + illiquid/idiosyncratic)
SAMPLE_NAMES = ["AAPL", "NVDA", "JPM", "XOM", "WMT", "KO", "PG", "CAT",
                "GME", "AMC", "SIRI", "F", "T", "PARA", "LUMN", "NWL"]


def _load(sym):
    try:
        d = pd.read_parquet(f"{PRICE_DIR}/{sym}.parquet")
    except Exception:
        return None
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def main():
    mem = SP500Membership.load()
    factors = {f: _load(f) for f in MACRO_FACTORS}
    missing = [f for f, v in factors.items() if v is None]
    if missing:
        raise SystemExit(f"missing macro factor data {missing} — warm it first")
    asof = min(v["date"].max() for v in factors.values())
    members = mem.members_asof(asof)

    syms = [(s, True) for s in V1_ETFS] + [(s, False) for s in SAMPLE_NAMES]
    # first pass for z-context (cross-sectional mean/sd of log $-vol and macro R²)
    prelim = []
    for s, is_etf in syms:
        px = _load(s)
        if px is None:
            continue
        t = score_symbol(s, px, factors, asof, is_member=(s in members), is_etf=is_etf)
        prelim.append((s, is_etf, px, t))
    dv = np.array([np.log10(max(t.dollar_vol, 1)) for *_, t in prelim if np.isfinite(t.dollar_vol)])
    r2 = np.array([t.macro_r2 for *_, t in prelim if np.isfinite(t.macro_r2)])
    z_ctx = {"dv": (dv.mean(), dv.std() or 1), "r2": (r2.mean(), r2.std() or 1)}

    rows = []
    for s, is_etf, px, _ in prelim:
        t = score_symbol(s, px, factors, asof, is_member=(s in members), is_etf=is_etf, z_ctx=z_ctx)
        rows.append(t)

    print(f"SWING SCANNER — Layer 0 TRADEABILITY GATE   as-of {asof.date()}")
    print(f"macro factors {MACRO_FACTORS}  |  GEX/options = SEAM (no OPRA data, flagged not faked)\n")
    print(f"{'symbol':8}{'kind':7}{'$vol/d':>9}{'macroR²':>9}{'flow':>6}{'score':>7}  {'verdict'}")
    for t in sorted(rows, key=lambda x: -x.score):
        kind = "ETF" if t.symbol in V1_ETFS else ("member" if t.index_flow else "name")
        verdict = "TRADEABLE" if t.tradeable else "reject: " + "; ".join(t.reasons)
        print(f"{t.symbol:8}{kind:7}{t.dollar_vol/1e6:>8.0f}M{t.macro_r2:>9.2f}"
              f"{'yes' if t.index_flow else 'no':>6}{t.score:>7.2f}  {verdict}")

    rejected = [t for t in rows if not t.tradeable]
    print(f"\n=== {len(rejected)} REJECTED (the gate working) — reasons ===")
    for t in rejected[:8]:
        print(f"  {t.symbol:6} -> {'; '.join(t.reasons)}")
    n_seam = len(rows[0].features.seams()) if rows else 0
    print(f"\nSEAMS surfaced per symbol: {n_seam} (dealer_gex, options_liquidity — pending OPRA). "
          f"Tradeable set: {sum(t.tradeable for t in rows)}/{len(rows)}.")


if __name__ == "__main__":
    main()
