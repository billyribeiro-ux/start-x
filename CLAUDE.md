# Project memory — Start-X swing-trading research

## Analysis window (LOCKED)
- **Default backtest / study window: Jan 1 2019 → June 2026** ("the latest data we use for now").
  Do NOT silently use 2010, 2015, or 2018 starts. If a longer history is ever needed for sample
  size, say so explicitly and still report the **2019→now** slice as the headline.
- Primary symbol: **SPY** (use SPX/^GSPC only for extra sample size). Horizon: short-term swing
  **1–10 days** (avoid intraday — it's noise for our purpose).

## Three separate books — one per horizon (LOCKED architecture)
The desk runs **three distinct systems, never mixed in one book** (mixing horizons under one exit
rule is what produced the 134-bar hold inside a "swing" book). Same 1-ATR-stop / 3-ATR-chandelier
rule everywhere; only the **max-hold clock = the horizon** differs. Get ONE right before combining.
1. **Short-term swing** — 1–10 trading days (IBS<0.1 dip-buy). HARD 10-day cap. ← System #1, locked.
2. **Long-term swing** — weeks to ~3 months (breakout + fear capitulation). 63-day cap. ← System #2,
   locked & OOS-validated (4-agent drill: real but FRAGILE — deflated Sharpe 0.81–0.86, large-cap only).
3. **Position / portfolio** — a long hold, months-to-years (200-SMA trend core). Multi-year backstop.
   ← System #3, scaffolded, NOT yet calibrated (next).
`scripts/run_book.py --book {short_swing,long_swing,position}`.

## Hard rules the user has set
- **Success = out-of-sample robustness**, never in-sample win rate or in-sample $.
- **Don't cut home-runners — *within a book's horizon*.** Ride winners to exhaustion via the 3-ATR
  chandelier, but a short-swing winner is still capped at 10 days, a long-swing one at ~3 months.
- **CSV trade-sheet layout is LOCKED** — do not keep changing columns. Include a totals block
  (TOTAL WIN $, TOTAL LOSS $, NET TOTAL $) and per-share P&L net of ~2bp SPY cost.
- A **+1 ATR or breakeven exit is a WIN/SCRATCH, not a LOSS.**
- Secrets (FMP key) live only in gitignored `.env`. Never commit secrets or generated CSVs.

## Settled findings (see FINDINGS.md for evidence)
- `prob_up` ML directional model is a coin flip OOS (AUC ~0.50). **Abandoned for production** — but
  still wired into the `forward/paper.py` research/forward harness (flagged there as research-only,
  NOT the validated system; the validated system is the three books in `scripts/run_book.py`).
- **IBS<0.1 dip-buy in an uptrend is the validated OOS mean-reversion edge** (System #1, short_swing).
  The legacy **RSI-2 (Connors) entry is DEPRECATED / superseded** — it decayed OOS (drift-adjusted
  alpha −0.22%, it was just riding the bull). Module: `strategy/mean_reversion.py` (IBS is the default;
  RSI-2 retained only as `signal="rsi2"`).
- **VIX "high = short the index" is FALSE on the swing horizon** — forward S&P rises with VIX; the
  −0.75 relationship is same-day/coincident, not predictive. VIX neutral price ≈ 18.
- **High VIX = capitulation LONG, not a short** (≥3 *consecutive* closes above the 2.5σ band,
  VIX>neutral → next-day gap-up). But be precise about status: the standalone
  `strategy/vix_capitulation.py` trigger is a **thin, high-conviction research edge** (deep-history
  7 trades, 71% win, PF 6 after the 2026-06 off-by-one fix) — REAL but small-sample, **not bankable
  alone and NOT wired into any production book**. The PRODUCTION `fear` sleeve (System #2 long_swing)
  uses the **better-sampled VVIX trigger** (`volatility_premium.fear_signals`); the spot-VIX
  capitulation trigger deflated to noise (DSR 0.10) and was replaced by VVIX.
