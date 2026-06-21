# STRATEGY — exact logic of every trade

This is the complete, plain-English specification of the production desk. **Nothing here is implicit.**
For each sleeve you get the precise ENTRY criteria, the EXIT logic (stop, target, time cap), how the
position is SIZED, and why the edge exists. Every exported trade also carries `entry_rule`,
`exit_rule`, `stop_price`, `stop_pct`, `regime`, `conviction`, and `thesis` columns so the row is
self-documenting.

- **Universe / horizon:** SPY primary (^GSPC/QQQ for extra sample). Study window **Jan 2019 → Jun
  2026** (locked). Costs ~2 bp/side. Headline stats are reported **over the study window** (the engine
  slices the equity curve to `[start, end]` before computing Sharpe/maxDD — folding flat pre-window
  days in understated Sharpe ~2×).

## Three separate books — one per horizon (LOCKED architecture)

The desk runs **three distinct systems, never mixed in one book.** Same 1-ATR-stop / 3-ATR-chandelier
rule everywhere; only the **max-hold clock = the horizon** differs. `scripts/run_book.py --book {…}`.

| Book | Horizon | Sleeves | max hold | Status |
|---|---|---|---|---|
| **short_swing** | 1–10 trading days | `ibs` | 10 d | ✅ #1 locked & OOS-validated |
| **long_swing** | weeks to ~3 months | `breakout` + `fear` | 63 d | ✅ #2 locked & OOS-validated (rated *real but FRAGILE*) |
| **position** | months to years | `breakout` (200-SMA core) | 504 d | ⏳ scaffolded, not yet calibrated |

**Models.** Only `short_swing` runs two models (`--model base|guarded|both`): **base** = IBS dip
ungated, **guarded** = IBS dip also requires healthy breadth (no falling knives). `long_swing` and
`position` carry no breadth-gated sleeve, so they expose a single `base` model (a "guarded" run would
be byte-identical).

---

## The exit rule (applies to every sleeve)

> **3-ATR chandelier to ride the winner, 1-ATR hard stop to cut the loser, plus a per-book maximum hold.**

Mechanically, each open trade tracks two levels off ATR(14) at entry:
- **Hard stop** = `entry − 1×ATR` — the floor; cuts the loss short.
- **Chandelier** = `(highest high since entry) − 3×ATR` — trails up as the trade runs.
- The **binding stop is the higher of the two.** At entry the 1-ATR hard stop binds; once the trade
  has run more than ~2 ATR in profit, the 3-ATR trail lifts above the hard stop and takes over
  (so a winner is given room, a loser is cut fast). Exit also forced at the **max-hold** cap.
- **"Don't cut home-runners" means within a book's horizon.** A short-swing winner is still capped at
  10 days; a long-swing winner at ~3 months. A trend that wants to ride 458 bars (e.g. the 2020 COVID
  rebound, uncapped) belongs to the **position** book — letting it run inside `long_swing` would
  re-create the "134-bar hold inside a swing book" horizon-mixing pathology the architecture forbids.

| Book → sleeve | hard stop | chandelier | max hold | why |
|---|---|---|---|---|
| short_swing → **ibs** | 1 ATR | 3 ATR | **10 d** | short-term dip — quick mean-reversion, never a year |
| long_swing → **breakout** | 1 ATR | 3 ATR | **63 d** | trend — ride weeks-to-~3-months (cap-swept: 63 is the spec-faithful plateau) |
| long_swing → **fear** | 1 ATR | 3 ATR | **63 d** | capitulation bounce — ride to exhaustion within the horizon |
| position → **breakout** | 1 ATR | 3 ATR | **504 d** | long hold — ride a trend for months-to-years |

---

## Sleeve — `ibs` (short-term oversold dip; System #1 = short_swing)

- **ENTRY:** **IBS < 0.10** — the close lands in the **bottom 10% of the day's range** — while SPY is
  **above its 200-day SMA** (an oversold dip inside an uptrend). IBS = (close − low) / (high − low).
- **GUARD (guarded model only):** skip the dip if it is a **broad breakdown** — up-volume ≤ 20% of
  total (≥80% down-volume across the S&P 500). Up/down volume separated IBS winners from losers at
  AUC 0.78; this vetoes falling knives (e.g. 2026-03-18).
- **EXIT:** 1-ATR hard stop, 3-ATR chandelier, **max 10 days** (it is a short-term trade by design).
- **Why it exists:** liquidity provision — buying the forced sell-into-the-close. Real, repeatable,
  but ~0 alpha over buy-and-hold; it earns its place as low-variance exposure timing + diversification.
- **Validated (2019-26):** guarded 29 trades, 59% win, +22.2%, PF 2.61, maxDD −3.6%, **in-window
  Sharpe 0.84**; base 41 trades, 51% win, +26.7%, PF 2.20, maxDD −6.3%, Sharpe 0.81. Positive in BOTH
  the 2019-22 bear train and the 2023-26 bull test; the guard wins win-rate/PF/drawdown in every regime.

## Sleeve — `breakout` (trend; the only sleeve with real drift-adjusted alpha; long_swing + position)

- **ENTRY:** SPY closes at a **new 20-day high** while **above its 200-day SMA**. (Buy strength
  confirming strength in an uptrend — a fresh breakout, not an extended one.)
- **EXIT:** 1-ATR hard stop, 3-ATR chandelier; **max 63 d in long_swing**, 504 d in position.
- **Why it exists:** you are paid by trend-followers/late longs chasing the breakout and by the
  disposition effect (early holders sell winners too soon). Validated OOS; generalizes to QQQ/^GSPC,
  **vanishes on small-caps (IWM)** — it's a large-cap/index phenomenon.

## Sleeve — `fear` (volatility-risk-premium long; the merged VRP ∪ VVIX trigger; long_swing)

- **ENTRY (fires on either):**
  - **VRP** = `VIX − 20-day realized vol` in the **top 5%** of its trailing year (fear richly priced
    vs how much the index is actually moving), **OR**
  - **VVIX** (vol-of-vol) **≥ its trailing-year 90th percentile** (dealer-hedging / convexity stress).
  - VRP and VVIX are two lenses on the same fear and are counted **once** (merged) so a vol spike
    books a single position, not two; the engine also refuses two positions at the same bar/price.
- **EXIT:** 1-ATR hard stop, 3-ATR chandelier, max 63 days (ride to exhaustion within the horizon).
- **Why it exists:** you get paid to provide liquidity / sell insurance into panic. Durable risk
  premium but **not alpha** over the index — a thin (19-trade) **time-coverage diversifier**: it fires
  ~2-3×/yr across all regimes and adds ~117 days the book would otherwise sit flat. Sized small.

### long_swing (System #2) — validated head-to-head (2019-26, SPY, cap 63)
54 trades, 46% win, **+87.4%**, PF 3.25, maxDD −10.1%, **in-window Sharpe 1.07**. Train (2019-22)
+24.9% / Sharpe 0.75; Test (2023-26) +50.0% / Sharpe 1.39 (OOS *stronger* than train, no decay).
**Not a COVID mirage:** drop 2020 entirely → still +66% / Sharpe 1.02; 6 of 8 years positive. Parameter
plateau is broad (27/27 perturbation cells positive — no knife-edge). **Honest ceiling:** deflated
Sharpe 0.81–0.86 — clears the 0.70 "fragile" floor but **below the 0.95 "REAL" bar**; top-3 trades are
~47% of P&L. Tradeable as a thin large-cap edge, sized accordingly — not a slam-dunk.

---

## Portfolio engine — sizing, overlays, hygiene

- **Concurrent, vol-targeted sizing.** Each position is sized so its risk ≈ **3% of equity** (sized
  on the chandelier distance, so the tighter 1-ATR hard stop only *reduces* realised loss). Caps:
  concurrent summed risk ≤ **12%**, summed gross ≤ **1.5×** (default; `--gross-cap`). 1.5× is the
  Calmar sweet spot; Sharpe is invariant to leverage (3× just amplifies P&L and deepens drawdown).
- **Reading the journal (R-multiples).** Every trade carries `risk_pct` (the 1-ATR risk distance,
  the R unit) and `R` (realised return in R units). The book's shape: **lose ~1R, win big, win
  ~half the time.** Gap-throughs are the only losers worse than −1R (worst −2.84R, the COVID gap).
- **Execution realism.** Entries fill at the signal-day close by default (`--entry-fill`); the book's
  edge SURVIVES next-open fills at the portfolio level (it's trend-capture, not entry precision) —
  but per-trade the entry signals have no standalone edge once you can't trade the close print.
- **Gold-calm overlay.** No NEW entries on days **GLD/SPY is above its 20-day SMA** (gold leading the
  index = risk-off / flight-to-safety). The single biggest out-of-sample risk improver.
- **Drawdown circuit-breaker.** While equity is below **peak × (1 − 10%)**, the weight of new entries
  is **halved** until the high-water mark is reclaimed.
- **No churn / no stacking.** A sleeve cannot re-enter the bar it exited; no two positions may open at
  the same entry price on the same bar (kills the same-day same-price duplication).
- **In-window stats.** Sharpe / total_return / maxDD / exposure are computed on the `[start, end]`
  slice (rebased), not the full price index — so they describe the period actually traded.

## How to run it

```
# System #1 — short-term swing (1-10 days), both models, regime stress grid
python scripts/run_book.py --book short_swing --start 2019-01-01 --end 2026-06-19 --model both --stress --out s.csv

# System #2 — long-term swing (weeks to ~3 months)
python scripts/run_book.py --book long_swing --start 2019-01-01 --end 2026-06-19 --stress --out l.csv
```
Writes one CSV per model plus `<out>_all.csv` (every trade tagged by `model`), each ending with the
locked totals block (TOTAL WIN $ / TOTAL LOSS $ / NET TOTAL $) and per-share P&L. Prints the
head-to-head and the regime stress grid. Each row is fully self-documenting.
