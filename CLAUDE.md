# Project memory — Start-X swing-trading research

## Analysis window (LOCKED)
- **Default backtest / study window: Jan 1 2019 → June 2026** ("the latest data we use for now").
  Do NOT silently use 2010, 2015, or 2018 starts. If a longer history is ever needed for sample
  size, say so explicitly and still report the **2019→now** slice as the headline.
- Primary symbol: **SPY** (use SPX/^GSPC only for extra sample size). Horizon: short-term swing
  **1–10 days** (avoid intraday — it's noise for our purpose).

## Hard rules the user has set
- **Success = out-of-sample robustness**, never in-sample win rate or in-sample $.
- **Don't cut home-runners.** Ride winners to exhaustion; the chandelier 3 ATR is the chosen exit.
- **CSV trade-sheet layout is LOCKED** — do not keep changing columns. Include a totals block
  (TOTAL WIN $, TOTAL LOSS $, NET TOTAL $) and per-share P&L net of ~2bp SPY cost.
- A **+1 ATR or breakeven exit is a WIN/SCRATCH, not a LOSS.**
- Secrets (FMP key) live only in gitignored `.env`. Never commit secrets or generated CSVs.

## Settled findings (see FINDINGS.md for evidence)
- `prob_up` ML directional model is a coin flip OOS (AUC ~0.50). Abandoned.
- RSI-2 dip-buy in an uptrend (Connors) is a validated OOS edge. Module: `strategy/mean_reversion.py`.
- **VIX "high = short the index" is FALSE on the swing horizon** — forward S&P rises with VIX; the
  −0.75 relationship is same-day/coincident, not predictive. VIX neutral price ≈ 18.
- **Persistent VIX vol-breakout (≥3 closes above the 2.5σ band, VIX>neutral) = a capitulation
  LONG.** Market gaps up ~+0.28% next day (~10x baseline). Module: `strategy/vix_capitulation.py`.
