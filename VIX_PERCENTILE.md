# VIX — the percentile / price → forward-return study

**The settled answer to "is high VIX a short or a long?"** Built on **8,143 trading days of overlapping
SPY + VIX history (1994–2026)**. Evidence files: `spy_vix_percentile_buckets.csv`,
`spy_vix_price_buckets.csv`, `spy_vix_capitulation_trades.csv` (all regenerable).

---

## TL;DR (the two punchlines)

| Relationship | Value | Meaning |
|---|---|---|
| **Coincident** — corr(VIX %chg, SPY same-day return) | **−0.71** | The famous "VIX up = market down." REAL but **simultaneous**, not predictive. |
| **Forward** — corr(VIX level, *next-10-day* SPY return) | **+0.06** | **Positive.** High VIX does NOT forecast a drop — if anything it precedes *higher* returns. |

> **High VIX (high percentile AND high price) is a forward LONG, not a short.** The −0.71 people cite
> is purely the same-day mechanical relationship. Across the whole history there is **no VIX level
> where forward drift turns negative** — the index drifts up at essentially every VIX bucket.

Unconditional forward drift (the baseline every bucket is measured against): **5d +0.20% · 10d +0.39% · 21d +0.82%.**

---

## Table A — VIX *percentile* (trailing-1yr, point-in-time) → forward SPY return

Higher percentile = more fearful/oversold relative to the recent year.

| VIX %ile bucket | n | VIX med | fwd-5d | fwd-10d | fwd-21d | fwd-10d win% |
|---|---|---|---|---|---|---|
| 0–10 (calm) | 1408 | 14.7 | +0.03 | +0.20 | +0.41 | 60% |
| 10–20 | 823 | 15.4 | +0.13 | +0.38 | +0.66 | 60% |
| 20–30 | 673 | 15.9 | +0.11 | +0.22 | +0.61 | 58% |
| 30–40 | 690 | 16.8 | +0.25 | +0.26 | +0.37 | 60% |
| 40–50 | 658 | 17.5 | +0.21 | +0.19 | +0.50 | 59% |
| 50–60 | 670 | 17.8 | +0.30 | +0.67 | +1.28 | 67% |
| 60–70 | 680 | 18.4 | +0.19 | +0.48 | +1.16 | 64% |
| 70–80 | 726 | 20.5 | +0.10 | +0.18 | +0.44 | 56% |
| 80–90 | 705 | 21.0 | +0.02 | +0.38 | +0.81 | 58% |
| **90–100 (fear extreme)** | 1110 | **25.7** | **+0.59** | **+0.86** | **+1.85** | **65%** |

**The top decile (most fearful) has the BEST forward returns of any bucket** (+0.86% / 10d, +1.85% / 21d,
65% up). Fear-percentile is a long signal. The calm bottom decile is the weakest.

---

## Table B — VIX *price* level → forward return (+ the percentile each price maps to)

This is the price-vs-percentile distinction: the same VIX *price* is a different *percentile* depending on
the regime, which is exactly why percentile ≠ price.

| VIX price | n | median %ile | fwd-5d | fwd-10d | fwd-21d | fwd-10d win% |
|---|---|---|---|---|---|---|
| < 12 | 701 | 10 | +0.15 | +0.36 | +0.74 | 66% |
| 12–15 | 1857 | 35 | +0.10 | +0.22 | +0.66 | 61% |
| 15–18 | 1548 | 47 | +0.17 | +0.38 | +0.64 | 62% |
| 18–20 | 856 | 43 | +0.14 | +0.36 | +0.62 | 61% |
| 20–25 | 1674 | 48 | +0.15 | +0.24 | +0.34 | 56% |
| 25–30 | 805 | 75 | +0.39 | +0.67 | +1.41 | 61% |
| 30–40 | 495 | 91 | +0.39 | +0.78 | +2.31 | 62% |
| **40+** | 207 | 94 | **+0.90** | **+1.45** | **+2.86** | 63% |

The highest *prices* (30–40, 40+) give the **highest forward returns**. And the `median %ile` column is the
bridge: VIX 12–15 ≈ 35th percentile, 25–30 ≈ 75th, 30–40 ≈ 91st.

---

## VIX "neutral price"

| Window | Median VIX (the attractor) |
|---|---|
| Full 1994–2026 | **17.9** |
| 2019–2026 | **18.2** |
| 2024–2026 | **16.6** |

The **~18 attractor** holds (the baseline crept up from the old $12–16 era). Honest nuance: because forward
drift is positive at *every* VIX level, "neutral" is **not** a drift-zero point — it's the **central
tendency (≈ 18)** that VIX oscillates *around*. Above it you're drifting into capitulation-long territory;
below it, complacency.

---

## The capitulation trigger (the actual entry built from this)

**Rule:** go LONG the index on the day VIX logs its **3rd consecutive close above its upper 2.5σ Bollinger
band (20-day) while VIX > neutral** — i.e. a *persistent* vol breakout, not a one-day spike. Exit: 3-ATR
chandelier, 40-day cap. Module: `strategy/vix_capitulation.py`. Trades (^GSPC, 1990–2026):

| Entry | VIX | Exit | Reason | $/sh | Outcome |
|---|---|---|---|---|---|
| 1992-10-05 | 20.0 | 1992-12-01 | time | +23.13 | WIN |
| 1997-10-29 | 33.8 | 1997-12-26 | time | +17.12 | WIN |
| 2001-02-22 | 26.8 | 2001-03-12 | chandelier | −47.31 | LOSS |
| 2014-12-12 | 21.1 | 2015-01-05 | chandelier | +34.14 | WIN |
| 2015-08-24 | 40.7 | 2015-09-22 | chandelier | +36.03 | WIN |
| **2020-10-28** | 40.3 | 2020-12-24 | time | **+431.38** | WIN |
| 2022-01-21 | 28.9 | 2022-02-14 | chandelier | −14.40 | LOSS |

**n=7 · 71% win · PF 6.0 · NET +$480/share.** REAL but **small-sample** — 7 deep-history trades. So:

> **Production status (be precise):** the standalone spot-VIX capitulation trigger is a **thin,
> high-conviction RESEARCH edge — NOT wired into any production book.** When deflated for sample size it
> drops to DSR ~0.10. The PRODUCTION `fear` sleeve (System #2 long_swing) instead uses the
> **better-sampled VVIX trigger** (`volatility_premium.fear_signals`), which replaced spot-VIX
> capitulation. The thesis (high VIX = capitulation long) is what's robust; this specific 7-trade entry is
> directional confirmation, not a bankable standalone system.

---

## Why the old "high VIX = short" intuition is wrong

It conflates the **coincident** −0.71 (VIX spikes *as* the market falls — true, same bar) with a
**predictive** claim (VIX is high → market will fall next — false, +0.06). Confirmed across VIX **level,
percentile, the 2.5σ bands, and term structure**: every fear extreme is a **bounce**, not a short. The one
asymmetry worth respecting is that the bounce is *persistent-fear* conditional — a single spike is noise;
the edge is in the 3rd-consecutive-close (capitulation) pattern.
