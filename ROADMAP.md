# Start-X — Roadmap / Backlog

Ideas captured to revisit. Nothing here is dropped — we fix/build when we get to it.

## Built ✅
- Data spine (rate-limited FMP client, Parquet cache), causal attribution engine
- 59 point-in-time features (technical, flow, regime, intermarket, % off ATH/ATL/recent H/L)
- Triple-barrier labels (long & short), pooled + per-symbol LightGBM, Optuna-under-purged-CV
- Validation scoreboard: purged CV / CPCV / deflated Sharpe / PBO
- Realistic backtest + walk-forward equity, forward/paper-tester (live blotter)
- Reversal Lab (ATR-zigzag + intraday POC/VWAP/RVOL/cum-delta + exit distributions)
- Evidence miner (rank what matched most across years of moves)
- Trade-blotter CSV export with target/stop/exit-reason

## In progress 🔨
- **Self-learning loss-autopsy + meta-label filter**: learn the fingerprint of losing trades
  from forensics (RVOL/POC/VWAP/cum-delta/ATR), retrain walk-forward, filter out setups that
  look like past losers, drift monitor triggers re-learn. (The engine behind the tune-up button.)

## Next / backlog 📋
- **"Tune-up" button (dashboard):** one click re-fits the models to current market conditions
  (UI trigger on the drift/walk-forward retrain loop).
- **Mean-reversion edge on SPY** (RSI-2 / oversold-bounce) — the one crack that pointed at real
  alpha (price rose 64% of the time the model screamed "oversold").
- **Evidence-based asymmetric stops/targets** — set stop just beyond measured adverse excursion,
  target near measured exhaustion (from Reversal Lab distributions); replace symmetric barriers.
- **Realistic per-instrument costs** — SPY slippage ~1bp, not the generic 5bp.
- **Intermarket lead detector** (rolling): which series leads SPY sell-offs *now* (yields,
  homebuilders, credit) and flag when the lead rotates.
- **Market internals scanner** ($TICK, $TRIN, $ADD, breadth, up/down volume) — pending a data
  provider (FMP doesn't carry these). Architecture already has a clean seam for it.
- **Catalyst-driven movers universe** (small/mid-caps) — where catalyst edges are far stronger
  than on the hyper-efficient S&P.
- **Window-independent feature cache** — cache full-history matrix per symbol/version and slice,
  so changing the date window never triggers a recompute (perf).
- **Svelte5/SvelteKit frontend** over a FastAPI layer exposing the same Python services.

## Honest guardrails (do not "fix" these away)
- Success = out-of-sample robustness, not win rate. A ~100% backtest is an overfitting/leak flag.
- The deflated Sharpe must be penalized for the number of configs tried (multiple testing).
- Beating buy-and-hold on the S&P is genuinely hard; "EDGE SURVIVES" ≠ "beats the index".
