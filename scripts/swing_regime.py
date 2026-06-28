"""Swing scanner — P2 entry: regime layer + classifier, with the charter's checkpoint.

Checkpoint (per the charter, "After P2: classify the last 24 months into regimes; eyeball against known
events (COVID, 2022 bear, OPEX clusters). Sanity or stop."). We classify the full real-data history so the
KNOWN events (2020 COVID, 2022 bear) can be checked — they're outside the last-24mo window — and also print
the last-24-month timeline. Regime is a conditioner, validated by eyeball, not by the Sharpe firewall.

    python scripts/swing_regime.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from startx.data.membership import SP500Membership
from startx.scanner.regime import _load, classify, regime_panel


def main():
    mem = SP500Membership.load()
    print("building regime panel (real data: vol-term/VVIX/MOVE/VRP/credit/dollar/trend + PIT breadth)...")
    panel = regime_panel(membership=mem)
    p = classify(panel).dropna(subset=["regime"])
    print(f"  classified {len(p)} days {p.index.min().date()}..{p.index.max().date()}\n")

    # forward 10d SPY return per regime (descriptive sanity — regime conditions, doesn't manufacture alpha)
    spy = _load("SPY")
    fwd10 = (spy.shift(-10) / spy - 1.0).reindex(p.index)
    print("=== regime distribution + forward 10d SPY (descriptive) ===")
    for r in ["risk_on", "neutral", "risk_off", "crisis"]:
        m = p["regime"] == r
        print(f"  {r:9} {m.mean()*100:5.1f}% of days  stress {p.loc[m,'stress'].mean():+.2f}  "
              f"fwd10d SPY {fwd10[m].mean()*100:+.2f}%  (n={int(m.sum())})")

    # CHECKPOINT — known-event validation
    print("\n=== CHECKPOINT: known events (the classifier must label these right) ===")
    events = {
        "2020-02-24..04-01 COVID crash": ("2020-02-24", "2020-04-01"),
        "2022-01-01..10-15 bear": ("2022-01-01", "2022-10-15"),
        "2018-12 Q4 selloff": ("2018-12-01", "2018-12-31"),
        "2017 calm melt-up": ("2017-01-01", "2017-12-31"),
        "2024-01..12 AI bull": ("2024-01-01", "2024-12-31"),
    }
    for label, (a, b) in events.items():
        seg = p[(p.index >= a) & (p.index <= b)]
        if seg.empty:
            print(f"  {label:32}: no data"); continue
        mix = (seg["regime"].value_counts(normalize=True) * 100).round(0).astype(int)
        dom = seg["regime"].value_counts().idxmax()
        mixs = " ".join(f"{k}:{v}%" for k, v in mix.items())
        print(f"  {label:32}: dominant={dom:9} stress {seg['stress'].mean():+.2f}  [{mixs}]")

    # last 24 months timeline (monthly dominant regime)
    print("\n=== last 24 months — monthly dominant regime ===")
    recent = p[p.index >= p.index.max() - pd.Timedelta(days=730)]
    mr = recent.groupby([recent.index.year, recent.index.month])["regime"].agg(
        lambda s: s.value_counts().idxmax())
    for chunk_start in range(0, len(mr), 6):
        items = list(mr.items())[chunk_start:chunk_start + 6]
        print("  " + "   ".join(f"{y%100:02d}-{mo:02d} {r}" for (y, mo), r in items))

    print("\nSEAMS left dark (no data): dealer gamma/charm/vanna, DIX, net-liquidity, implied correlation.")
    print("Regime is a CONDITIONER — eyeball the events above; it does not promote as a signal (P4 gates that).")


if __name__ == "__main__":
    main()
