# Start-X — Roadmap / Backlog

Ideas captured to revisit. Nothing here is dropped — we fix/build when we get to it.

## Recommendations — next improvements (2026-06-21, priority order)
Now that the desk is **three separate books** and **System #1 (short_swing) is locked & OOS-validated**,
these are the highest-leverage next moves, hardest-evidence first:

1. ~~**Calibrate System #2 — long_swing.**~~ ✅ DONE — locked at 63d, OOS-validated (real but FRAGILE,
   deflated Sharpe 0.81–0.86), 4-agent drill. See CHANGELOG / STRATEGY.md.
2. ~~**Calibrate System #3 — position (long hold).**~~ ✅ DONE — 200-SMA ±3% band trend core
   (`strategy/trend_position.py`), benchmarked vs SPY B&H: deep-cycle 1990-2026 ≈ the index's return
   at 38% of its drawdown (Calmar 0.39 vs 0.15). Honest verdict: drawdown defense, not a B&H-beater;
   lags in bull-only windows; leverage/re-entry rails proven not to help. **All three books now built.**
3. **Walk-forward the PARAMETERS, don't in-sample them.** The 10-day cap, IBS<0.1 threshold, and 1/3-ATR
   mults were picked on the full sample. Re-fit on a rolling train window, apply forward, and report a
   **deflated Sharpe penalized for the number of configs tried** — otherwise the cap/threshold choice is
   itself a quiet overfit.
4. **Add SPX/QQQ/IWM for sample size.** 29-41 short-swing trades over 7yr is thin. Confirm the IBS<0.1 +
   10-day-cap edge holds on SPX/^GSPC/QQQ/IWM (CLAUDE.md allows it for extra sample) — if it's SPY-only,
   it's luck. Report the **2019→now SPY slice as the headline** regardless.
5. **Emit per-trade excursion (MFE/MAE) from the engine.** The ledger can't yet show max favorable /
   adverse excursion, so we can't *prove* the 3-ATR chandelier beats a fixed target or set evidence-based
   stops from measured adverse excursion. Add `mfe_R`/`mae_R` to the engine ledger (real, not fabricated).
6. **Re-stream the ensemble on the THREE clean books.** The DECODE.md ensemble (Sharpe 1.38) was assembled
   on the OLD mixed book — its weights are stale. Re-run `portfolio/ensemble.py` over short_swing /
   long_swing / position (+ the market-neutral ML ranker, the best diversifier at corr ~0.09).
7. **Honest cost / execution per book.** Short_swing turns over far more than position, so cost bites
   harder there. Re-confirm the short edge survives **realistic SPY slippage (~1bp)** and **next-open
   fills** — we already found entry signals have ~no standalone edge off the close print, which caps any
   attempt to trade the short book more aggressively. Bake that constraint in, don't fight it.
8. **Regime-sizing overlay (not a new signal).** The validated K-means/HMM regime model puts the highest
   forward returns in stress/capitulation regimes — use it to size each book up in capitulation, down in
   calm. Sizing, not entries; keep it out of the entry logic.

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
- ~~**Mean-reversion edge on SPY** (RSI-2 / oversold-bounce)~~ — SHIPPED & superseded: RSI-2 decayed
  (drift-adjusted alpha −0.22%); the surviving oversold edge is **IBS<0.1**, now System #1 (short_swing).
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
