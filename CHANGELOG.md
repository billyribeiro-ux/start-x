# Changelog

All notable changes to the Start-X swing-trading research engine. Newest first.
Discipline throughout: **out-of-sample** robustness (TRAIN 2019-22 / TEST 2023-26),
drift-adjusted alpha, deflated Sharpe, brutal honesty about multiple testing and small samples.
Evidence lives in `FINDINGS.md`; the locked study window is **Jan 2019 → Jun 2026** (`CLAUDE.md`).

## 2026-06-21

### Added / Changed (parallel drill — 3 Opus agents)
- **Leverage drill → default gross_cap 3.0×→1.5×.** Sharpe is invariant to leverage (1.18→1.21 from
  1×→3×) — leverage only amplifies. maxDD scales sub-linearly and the cap rarely binds (mean
  in-market gross 0.74→1.07× as cap 1→3; ~90% of days ≤1.0×). 1.5× is the Calmar peak; 1.0× is the
  un-levered product: **8.8% CAGR / −9.1% maxDD / Sharpe 1.18 / Calmar 0.96 vs SPY 15.8% / −34.1% /
  0.85 / 0.46** — half the return, a quarter of the drawdown. `--gross-cap` flag (default 1.5).
- **Entry-fill realism (`--entry-fill close|next_open`).** Book edge SURVIVES next-open execution
  (Sharpe/PF/maxDD ~unchanged, OOS too) — it's trend-capture, not entry precision. BUT every sleeve
  buys a +3–8 bp overnight gap-up, so per-trade drift-adjusted alpha goes negative under next-open
  (IBS −0.16% t=−2.01; breakout mildly negative). Entry signals have no standalone edge off the
  close print — caps any attempt to scale entry frequency / shorten holds.
- **R-multiple journal columns** (`risk_pct`, `R`) on every trade. Expectancy **+1.12R (base) /
  +1.28R (guarded)**; avg win +3.4R, avg loss −1.0R; losers cluster at −1R, gap-throughs the only
  tail past it (worst −2.84R, the COVID gap). `mfe_R`/`mae_R` not added — engine ledger doesn't emit
  excursion (honest skip, not fabricated).
- **Final book @ 1.5× (both models, 2019-26):** base +111.8% / PF 3.05 / −12.8% / Sharpe 0.77;
  guarded +110.2% / PF 3.51 / −11.3% / Sharpe 0.79.

### Fixed (realism — ledger audit)
- **Gap-aware fills.** A ledger audit found 13 trades filling OUTSIDE the exit day's range: stops
  were filled at the exact stop level even when the bar GAPPED through it, overstating winners and
  hiding the tail (the 2020-02-24 COVID gap looked like -1.0% but was really -2.9%). Now fills at
  `min(stop_level, open)`. Honest re-baseline FULL 2019-26: base +120.5% / PF 2.72 / maxDD -15.2%;
  guarded +135.4% / PF 3.22 / maxDD -13.4% / Sharpe 0.79. Worst single loss now -2.94%.

### Fixed (ledger drill)
- **Gap-aware fills** (above).
- **Fear sleeve no longer cuts its runner.** Drill found the fear 90-day cap time-exited the +29.8%
  COVID capitulation winner (same pattern as the breakout time-cap). Fear cap 90→252 (chandelier is
  the real exit): FULL 2019-26 +120.5%→+127.7%, PF 2.72→2.88, Sharpe 0.73→0.76, same -15.2% maxDD.
- **Noted (by design, not a bug):** peak concurrent GROSS leverage reaches the 3.0× cap — the book
  can be up to ~3× long SPY at once (amplifies both return and drawdown); risk is held by the 1-ATR
  stops + 12% summed-risk cap (realised maxDD -15%).

### Changed (exit logic — user-directed)
- **Per-sleeve exits + the explicit stop/target rule** (user-found): the engine was applying ONE
  exit (3-ATR stop, 252-day cap) to every sleeve regardless of horizon — wrong for the short-term
  IBS dip (it got a 3.25% / $23 stop and a one-year hold). Now each sleeve carries
  `(hard-stop ATR, chandelier ATR, max-hold)`: **1-ATR hard stop to cut the loss, 3-ATR chandelier
  to ride the winner**, with breakout 252d / fear 90d / **IBS 10d**. The 2026-05-01 IBS stop went
  from 697.26 (3 ATR) to **712.85 (1 ATR, −1.08%)**; IBS holds are capped at 10 days while breakout
  still rides 134. Realised loss per trade is now ≤1 ATR (lower drawdown).

### Added (transparency — user-directed)
- **Every trade is self-documenting.** Ledger now carries `entry_rule`, `exit_rule`, `stop_price`,
  `stop_pct` columns, and a full spec doc **`STRATEGY.md`** gives the exact ENTRY/EXIT/SIZING logic
  of every sleeve in plain English. ("I shouldn't have to ask" — now you don't.)
- **Two models kept side by side** (`scripts/run_book.py --model base|guarded|both`, default `both`)
  so the breadth guard is judged head-to-head, not by assertion: **base** = IBS dip ungated;
  **guarded** = IBS dip breadth-guarded. `--out` writes one CSV per model; a head-to-head table
  prints win/PF/maxDD/Sharpe for both. (Per user: race both, log every change.)
- **Stress grid** (`--stress`): both models across 2019-22 (incl. bear) / 2023-26 (bull) / full, so
  regime-dependence is visible at a glance. Result: **guarded wins the 2019-22 bear outright** (win
  50→53%, PF 2.74→3.16, maxDD −17.7→−14.8%) — the guard is a stress protector; base edges raw return
  in the calm bull. **Combined all-trades export** (`<out>_all.csv`): every trade from every model,
  tagged by `model`.
- **Market-internals layer** (`strategy/market_internals.py`). Derives 5 breadth internals from the
  503 S&P-500 constituents (no vendor breadth feed on the FMP plan): % above 50/200d MA,
  advance/decline, up/down volume, new highs−lows, McClellan. Point-in-time (trailing quantiles,
  `dateFirstAdded` membership gate); cached to `data/cache/internals.parquet`.
- **IBS breadth-guard** (`not_breaking_down`): the IBS dip sleeve now stands aside on heavy
  down-volume "breakdown" days (up-volume ≤ 20% / ≥80% down-volume). Up/down volume separated IBS
  winners from losers at **AUC 0.78**. Effect (risk filter, not alpha): win-rate 58→61%, **max
  drawdown −19.4%→−15.5%** on FULL 2019-26; vetoes the 2026-03-18 falling-knife loss. Small raw-
  return give-up on TEST (honest).
- **`fear` sleeve** = merged VRP ∪ VVIX (`volatility_premium.fear_signals`), counted once.
- **Production portfolio package** (`portfolio/`): `engine` (concurrent vol-sized book, chandelier
  exit, gold-calm overlay, drawdown circuit-breaker), `thesis` (per-trade regime/conviction/causal
  PM thesis + breadth read), `execution` (realistic fills), `validate` (deflated Sharpe / PBO /
  drift-adjusted-alpha scorecard, paper-forward). Runner: `scripts/run_book.py`.

### Changed
- **Default mean-reversion entry RSI-2 → IBS<0.1** — RSI-2 decayed (drift-adjusted alpha −0.22%, it
  was just riding the bull); IBS<0.1 is the only oversold survivor across SPY/SPX/QQQ/IWM.
- **VIX neutral is now adaptive** (trailing zero-drift attractor ~18, was a static 18).
- **VRP+VVIX merged into one fear sleeve** so a vol spike books one position, not two.

### Fixed
- **Time-cap churn** (user-found): a 40-day clock force-exited still-trending winners and the engine
  re-bought the same bar/price. Chandelier is now the real exit (cap → 252 backstop) + no same-bar
  re-entry. Book FULL 2019-26 +225%→+271%, PF 3.28→4.46, maxDD −21.4%→−18.8%.
- **Same-bar same-price stacking** (user-found): three sleeves booked the identical 2026-03-18 trade
  (−4%×3). Engine now refuses two positions at the same entry price on one bar.
- **Leakage in `learning/forensic_feature_columns`**: deny-list let `ret`/`exit_price`/levels feed
  the meta-model (fake "100% win"). Switched to an allow-list + regression test.

### Findings / honest negatives (not shipped)
- **Breadth *confirmation* for the fear sleeve does NOT generalize** (n=13; the 2026-03 example was a
  cherry-pick; gate slightly hurt OOS). Only the IBS breadth *guard* survived.
- **ML meta-filters & learned exits do not beat the rules** (purged-CV AUC <0.5, PBO 0.83); the
  earlier "84% win" IBS meta-filter was fragile (drawdown-smoothing only).
- **VIX-capitulation demoted to likely-noise** (deflated Sharpe 0.10, profit in 5 lucky trades);
  replaced by the better-sampled VVIX trigger inside the merged fear sleeve.
- **High VIX is not a short** — confirmed across level, percentile, 2.5σ bands, and VIX **term
  structure**. Fear extremes are bounces, not shorts.
