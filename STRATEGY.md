# STRATEGY — exact logic of every trade

This is the complete, plain-English specification of the production book. **Nothing here is implicit.**
For each sleeve you get the precise ENTRY criteria, the EXIT logic (stop, target, time cap), how the
position is SIZED, and why the edge exists. Every exported trade also carries `entry_rule`,
`exit_rule`, `stop_price`, `stop_pct`, `regime`, `conviction`, and `thesis` columns so the row is
self-documenting.

- **Universe / horizon:** SPY only. Study window **Jan 2019 → Jun 2026** (locked). Costs ~2 bp/side.
- **Two models, raced side by side** (`scripts/run_book.py --model base|guarded|both`):
  - **base** — IBS dip sleeve ungated.
  - **guarded** — IBS dip sleeve also requires healthy breadth (no falling knives).

---

## The exit rule (applies to every sleeve)

> **3-ATR chandelier to ride the winner, 1-ATR hard stop to cut the loser, plus a per-sleeve
> maximum hold.**

Mechanically, each open trade tracks two levels off ATR(14) at entry:
- **Hard stop** = `entry − 1×ATR` — the floor; cuts the loss short.
- **Chandelier** = `(highest high since entry) − 3×ATR` — trails up as the trade runs.
- The **binding stop is the higher of the two.** At entry the 1-ATR hard stop binds; once the trade
  has run more than ~2 ATR in profit, the 3-ATR trail lifts above the hard stop and takes over
  (so a winner is given room, a loser is cut fast). Exit also forced at the **max-hold** cap.

**Per-sleeve exit horizon** (`stop ATR`, `chandelier ATR`, `max hold`):

| Sleeve | hard stop | chandelier | max hold | why |
|---|---|---|---|---|
| **breakout** | 1 ATR | 3 ATR | **252 d** | long-term trend — ride it for months (the home-runner) |
| **fear** | 1 ATR | 3 ATR | **90 d** | capitulation bounce — weeks to a quarter |
| **ibs** | 1 ATR | 3 ATR | **10 d** | SHORT-TERM dip (1–10 days) — quick mean-reversion, never a year |

---

## Sleeve 1 — `breakout` (trend; the only sleeve with real drift-adjusted alpha)

- **ENTRY:** SPY closes at a **new 20-day high** while **above its 200-day SMA**. (Buy strength
  confirming strength in an uptrend — a fresh breakout, not an extended one.)
- **EXIT:** 1-ATR hard stop, 3-ATR chandelier, max 252 days. Rides trends for months.
- **Why it exists:** you are paid by trend-followers/late longs chasing the breakout and by the
  disposition effect (early holders sell winners too soon). Validated OOS; generalizes to QQQ/SPX.

## Sleeve 2 — `ibs` (short-term oversold dip; exposure timing)

- **ENTRY:** **IBS < 0.10** — the close lands in the **bottom 10% of the day's range** — while SPY is
  **above its 200-day SMA** (an oversold dip inside an uptrend). IBS = (close − low) / (high − low).
- **GUARD (guarded model only):** skip the dip if it is a **broad breakdown** — up-volume ≤ 20% of
  total (≥80% down-volume across the S&P 500). Up/down volume separated IBS winners from losers at
  AUC 0.78; this vetoes falling knives (e.g. 2026-03-18).
- **EXIT:** 1-ATR hard stop, 3-ATR chandelier, **max 10 days** (it is a short-term trade by design).
- **Why it exists:** liquidity provision — buying the forced sell-into-the-close. Real, repeatable,
  but ~0 alpha over buy-and-hold; it earns its place as low-variance exposure timing + diversification.

## Sleeve 3 — `fear` (volatility-risk-premium long; the merged VRP ∪ VVIX trigger)

- **ENTRY (fires on either):**
  - **VRP** = `VIX − 20-day realized vol` in the **top 5%** of its trailing year (fear richly priced
    vs how much the index is actually moving), **OR**
  - **VVIX** (vol-of-vol) **≥ its trailing-year 90th percentile** (dealer-hedging / convexity stress).
  - VRP and VVIX are two lenses on the same fear and are counted **once** (merged) so a vol spike
    books a single position, not two; the engine also refuses two positions at the same bar/price.
- **EXIT:** 1-ATR hard stop, 3-ATR chandelier, max 90 days (let the rebound develop).
- **Why it exists:** you get paid to provide liquidity / sell insurance into panic. Durable risk
  premium but **not alpha** over the index — sized small, kept as a diversifier (low correlation to
  the trend sleeve).

---

## Portfolio engine — sizing, overlays, hygiene

- **Concurrent, vol-targeted sizing.** Each position is sized so its risk ≈ **3% of equity** (sized
  on the chandelier distance, so the tighter 1-ATR hard stop only *reduces* realised loss). Caps:
  concurrent summed risk ≤ **12%**, summed gross ≤ **3.0×**; a new trade scales down to fit or is
  skipped.
- **Gold-calm overlay.** No NEW entries on days **GLD/SPY is above its 20-day SMA** (gold leading the
  index = risk-off / flight-to-safety). The single biggest out-of-sample risk improver.
- **Drawdown circuit-breaker.** While equity is below **peak × (1 − 10%)**, the weight of new entries
  is **halved** until the high-water mark is reclaimed.
- **No churn / no stacking.** A sleeve cannot re-enter the bar it exited; no two positions may open at
  the same entry price on the same bar (kills the same-day same-price duplication).

## How to run it

```
python scripts/run_book.py --start 2019-01-01 --end 2026-06-19 --model both --stress --out book.csv
```
Writes `book_base.csv`, `book_guarded.csv`, and `book_all.csv` (every trade, both models, tagged by
`model`), prints the head-to-head and the regime stress grid. Each row is fully self-documenting.
