# Changelog

All notable changes to the Start-X swing-trading research engine. Newest first.
Discipline throughout: **out-of-sample** robustness (TRAIN 2019-22 / TEST 2023-26),
drift-adjusted alpha, deflated Sharpe, brutal honesty about multiple testing and small samples.
Evidence lives in `FINDINGS.md`; the locked study window is **Jan 2019 → Jun 2026** (`CLAUDE.md`).

## 2026-06-21

### Added
- **Two models kept side by side** (`scripts/run_book.py --model base|guarded|both`, default `both`)
  so the breadth guard is judged head-to-head, not by assertion: **base** = IBS dip ungated;
  **guarded** = IBS dip breadth-guarded. `--out` writes one CSV per model; a head-to-head table
  prints win/PF/maxDD/Sharpe for both. (Per user: race both, log every change.)
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
