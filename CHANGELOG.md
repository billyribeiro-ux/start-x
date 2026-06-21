# Changelog

All notable changes to the Start-X swing-trading research engine. Newest first.
Discipline throughout: **out-of-sample** robustness (TRAIN 2019-22 / TEST 2023-26),
drift-adjusted alpha, deflated Sharpe, brutal honesty about multiple testing and small samples.
Evidence lives in `FINDINGS.md`; the locked study window is **Jan 2019 → Jun 2026** (`CLAUDE.md`).

## 2026-06-21

### Built — System #3: position book (`strategy/trend_position.py`)
- **The third book is live: a 200-SMA ±3% hysteresis-band trend core**, long-above / flat-below, its
  OWN engine (a continuous regime allocation, NOT the discrete chandelier path). The "stop" is the
  trend breakdown (the SMA band) — there is deliberately no 1-ATR stop or chandelier, which would
  knock a months-to-years hold out within days. Point-in-time (state decided on the prior close).
  `scripts/run_book.py --book position` prints the head-to-head vs SPY buy-and-hold and writes the
  locked-layout ledger (`--band`/`--sma-len` to tune).
- **Honest mandate — drawdown defense, NOT a B&H-beater** (the 4-agent calibration drill: plain
  timing has no Sharpe edge over B&H, leverage is no free lunch, re-entry rails don't robustly help):
  - **Deep cycle ^GSPC 1990-2026:** +8.6% CAGR / −21.7% maxDD / Calmar 0.39 / Sharpe 0.77 vs B&H
    +8.7% / −56.8% / 0.15 / 0.55 — **≈ the index's return at 38% of its drawdown** (Calmar 2.6×),
    cutting all four bears (dot-com, GFC, COVID, 2022) ~in half.
  - **Locked 2019-26 (bull-only, it lags by design):** +9.2% / −22.1% / Calmar 0.42 vs B&H +15.8% /
    −34.1% / 0.46 — keeps 58% of B&H CAGR at 65% of its drawdown; value is insurance against the
    −50%+ tail the window doesn't contain. 6 trades, the winners held 434/530 bars (months-to-years).
  - Tests: `tests/test_trend_position.py` (PIT stability, band hysteresis, ledger/benchmark, turnover).
  **All three books now built.** (STRATEGY.md + CLAUDE.md updated.)

### Forensic re-audit (6 parallel Opus agents) — verdict: no faked edge
A full-codebase hostile forensic sweep with reproducible probes. **The core is CLEAN:** the production
money path (engine/run_book) has no lookahead / cap-defeat / cost-double-charge (all recent fixes
re-verified PASS, in-window Sharpe 1.163 == manual); the signal sleeves are lookahead-clean (8/8
PIT-truncation probes); the feature matrix is point-in-time (shuffle-label canary AUC 0.489, planted
leak 0.989); and the validation firewall (purge/embargo, CPCV, deflated-Sharpe, PBO, feature
allow-list) is mathematically correct. **The System #1/#2 locks rest on sound foundations.** The
defects found are confined to (a) the price-cache coverage wiring, (b) the research-only causal-
attribution engine, and (c) the abandoned `prob_up` ML pipeline — none touch the three-book P&L.

### Fixed (forensic re-audit — strategy)
- **`not_breaking_down` no longer blocks trades on ABSENT internals.** Its contract is "missing → True
  (don't block)", but `np.nan > x` is False, so missing breadth silently BLOCKED the IBS dip (and the
  trailing `.fillna(True)` was a dead no-op). NaN now coerces to True before the comparison. 0 days
  affected in the locked 2019-26 window (internals dense there); 4,265 deep-history days were wrongly
  suppressed. Regression test added.
- **Breadth reads constituents under the cache's sanitized filename.** `compute_internals` read
  `prices/{sym}.parquet` raw while `get_prices` writes the cache-sanitized name — a dotted class share
  (`BRK.B` → `BRK_B.parquet`) would silently drop from breadth. Now mirrors the cache `_SAFE` regex.
- **`vix_capitulation.backtest` outcome now WIN/SCRATCH/LOSS** (was binary — the one sleeve helper
  missed in the earlier harmonization), matching `engine._ledger_row` and the locked rule.

### Fixed (forensic re-audit — ensemble)
- **`ensemble._deflated_sharpe` replaced by the audited Bailey/LdP formula.** The local version was
  mangled four ways (incl. dividing `sr_star` by `√ppy` twice) — a *pessimistic* bug that UNDERSTATED
  the DSR. It now delegates to `validation.metrics.deflated_sharpe`. Recomputed on the 89-month
  decoded ensemble: **DSR 0.61 → 0.995** (the prior 0.61 in DECODE.md is corrected; the ensemble
  actually clears the 0.95 "REAL" bar). Off the production money path, but it's the number DECODE.md
  reported. `tests/test_ensemble.py` added (7).
- **Ensemble max-Sharpe ridge made scale-aware** (`1e-10` → `1e-6·trace(cov)/n`) so the tangency
  solve doesn't explode on near-collinear edges under `long_only=False` (default long-only unchanged).

### Fixed (forensic re-audit — data layer)
- **Cache-coverage check wired into `get_prices` (HIGH).** The coverage/TTL logic in `cache.py` was
  DEAD CODE — `get_prices` never passed the coverage args, so a truncated/stale price cache was
  served verbatim regardless of the requested history. Now passes `coverage_start=history_start` +
  `coverage_end=<last expected TRADING day>` so a wider/stale request re-fetches; the trading-day
  comparison (not raw `today`) means a weekend/holiday run against an up-to-date cache does NOT
  falsely re-fetch. HTTP 429 added to the client's retryable set with bounded backoff (was swallowed
  into an empty frame). Tests added (`test_prices.py`, `test_client.py`; `test_cache.py` extended).

### Fixed (forensic re-audit — validation hardening; ML/research pipeline, not the books)
- **`walk_forward_predict` now PURGES label-span overlap** (not just a positional embargo) — train
  labels whose `[entry,t1]` reaches into the test block are dropped (the LdP purge the long/position
  label horizons needed).
- **`PurgedKFold` no longer crashes on the production `Dataset` shape.** It treated `X.index` (a
  RangeIndex from `dataset.py`) as datetime label-starts → `Int64 vs DateTime64` error, killing the
  Optuna OOS-AUC tuner; now derives starts from `t1`'s values. Regression test on the RangeIndex shape.
- **Feature allow-list closed the `candle_` hole.** A numeric `candle_*` column could pass blindly;
  now trusted only if a known candle archetype OR genuinely 0/1-valued. Probe test added.
- **The label-span purge now actually ENGAGES in the pipeline path** (follow-through on FX2's report).
  `pipeline._sorted_arrays` reset `t1` to a RangeIndex, discarding the entry-date index
  `walk_forward_predict` reads — silently downgrading the new purge back to a bare positional embargo.
  `t1` now keeps its sorted entry-date index.
- **`cpcv.py` twin RangeIndex crash fixed.** `CombinatorialPurgedCV.split` still used the raw
  `starts = index.to_numpy()` pattern (the same Int64-vs-DateTime64 bug `PurgedKFold` had); now uses
  the shared `label_start_end` helper for type-consistent label bounds.
- **Full `pytest` runs in one process again.** Root `conftest.py` pins the BLAS/OpenMP thread pools to
  1 (`OMP/OPENBLAS/MKL/NUMEXPR/VECLIB`), so the heavy LightGBM/SHAP modules no longer oversubscribe
  threads and get the process killed mid-run (the spurious "OOM"/exit-137). One-shot suite: **191
  passed / 5 skipped**.

### Fixed (forensic re-audit — causal-attribution engine; HIGH leakage)
- **Same-day POST-CLOSE catalysts no longer attributed to the move (HIGH).** `attribute.py` used a
  day-resolution window that credited post-16:00 news / analyst actions / AMC earnings to that day's
  move (fabricated causality — e.g. a post-close beat inflated confidence 0.0→0.70). A `SESSION_CLOSE`
  (16:00) gate now keeps at/before-close catalysts on their day and advances post-close ones to the
  NEXT trading day. Earnings with no intraday time default to AMC (`16:00:01` → t+1), the conservative
  PIT choice.
- **Evidence-Miner made truthful** — its catalyst flags respect the same post-close gate, so the page's
  "prior-day (t-1) snapshot" claim holds; lookback converted from calendar to trailing TRADING days (a
  Friday catalyst → Monday move is no longer dropped). Same-day-post-close test added.

### Fixed (the three flagged known-limitations — "get everything fixed")
- **`days_to_next_earnings` PIT leak closed.** `features/flow.py` counted days to the next future
  earnings date regardless of whether it was scheduled/announced at `t` — leaking a not-yet-known
  date (the docstring even claimed a publish-date gate the code never implemented). Now clipped to
  ~one quarter (`_MAX_SCHED_HORIZON_D=92`): a date >92d out → NaN. Docstrings reconciled; regression
  test added (far-future → NaN, within-quarter still counted).
- **Fresh-cache breadth/internals.** `backfill.py` warmed only the 7 seed symbols + context series,
  so a clean clone had no S&P 500 constituent prices and `compute_internals` (which silently skips
  missing files) produced EMPTY breadth — the guarded model's IBS sleeve would have no guard. backfill
  now warms the ~500 constituents and rebuilds `data/cache/internals.parquet` (with a `--no-breadth`
  escape for a fast warm).
- **Aux-script windows aligned to the locked 2019 start.** `validate.py` (2015→2019),
  `export_trades.py` (2018→2019), and `highconf_eval.py` `--train-start` (2018→2019) defaulted outside
  the locked study window; now consistent with the production runner (`run_book.py` already 2019).

### Audit confirmation round — end-to-end verification + the genuinely-open fixes
A second meta-audit was confirmed claim-by-claim against the CURRENT tree. Its corrections to the
first pass were re-verified TRUE: production does NOT violate the locked rules — `run_book.py` BOOKS
passes per-book `exits` (10/63/504; the 252 is only an engine fallback), `engine._ledger_row` does
WIN/SCRATCH/LOSS, `2010` is warm-up data and the engine clips trades to the 2019 window, `vol_target`
is tested, and `normalize_ledger` charges cost once. Binary WIN/LOSS lives only in research-helper
`backtest()` functions, not the production ledger. The genuinely-open items were fixed:
- **Doc contradictions reconciled (CLAUDE.md + module docstring).** RSI-2 is now stated as
  DEPRECATED/superseded by IBS<0.1 (was "a validated OOS edge", contradicting the code's DEPRECATED
  and FINDINGS/ROADMAP). The VIX-capitulation entry is now precise: a THIN (deep-history 7-trade)
  research edge, REAL but not bankable alone and **NOT in any production book** — the production
  `fear` sleeve uses the VVIX trigger (`volatility_premium.fear_signals`); the spot-VIX trigger
  deflated to noise (DSR 0.10) and was replaced. `vix_capitulation.py`'s "validated on 16y" header
  reconciled to the same status.
- **prob_up forward harness flagged honestly.** `forward/paper.py` still scores the ABANDONED
  `prob_up` directional model (OOS AUC ~0.50; trades both sides incl. SHORTS; no meta-label gate).
  Added a module-docstring banner + one-time RuntimeWarning (in `forward_test` and `live_signals`)
  and an `st.warning` on dashboard page `5_Forward_Paper.py`, marking it research-only — NOT the
  validated three-book system. Proper fix (repoint the harness to forward-test `run_book`'s three
  books, long-only) logged as a recommendation; `tests/test_forward.py` still green.
- **Labeling presets aligned to the three books (`labeling/config.py`).** `long` horizon 60→**63**
  (matches the long_swing cap), and a new `position` preset (252d) added — the 60-vs-63 drift removed.
- **Locked CSV totals made rule-consistent + tested.** `_write_ledger` had SCRATCH in neither
  TOTAL WIN $ nor TOTAL LOSS $, so NET omitted scratch P&L. Per the locked "+1-ATR/breakeven =
  WIN/SCRATCH" rule, scratch now sums on the WIN side and **NET TOTAL $ = every trade**. New
  `tests/test_run_book_ledger.py` locks the layout + totals math (incl. a scratch row). (Live SPY
  runs emit zero scratch today, so today's NET is numerically unchanged — but now correct by construction.)
- **Research sleeve helpers harmonized to WIN/SCRATCH/LOSS + first dedicated tests.** The standalone
  `backtest()` in `momentum_breakout.py` and `volatility_premium.py` classified outcomes binary
  WIN/LOSS (a breakeven counted as LOSS — violating the locked rule, though these are research
  helpers, not the production ledger). Both now use `>0 WIN / ==0 SCRATCH / <0 LOSS`, matching
  `engine._ledger_row`. Added `tests/test_momentum_breakout.py` (9) and `tests/test_volatility_premium.py`
  (11) — the two trusted-but-untested sleeves now have real coverage (signals + backtest smokes).
  Suite **141 → 162**.

### Fixed (correctness — the off-by-one the audit's new tests surfaced)
- **`vix_capitulation.capitulation_signals` fired one close early.** The run-length counter grouped by
  `(~above).cumsum()`, which folds the breaking False bar into the following run, so the FIRST
  consecutive above-band close was counted as 2 — `min_closes=3` fired on the 2nd close, and an
  *isolated single spike* counted as run==2 (would fire at `min_closes=2`, defeating the "persistent"
  thesis). Fixed to transition-based grouping (`(above != above.shift()).cumsum()`, False bars forced
  to 0) so the run counts 1,2,3,… from the first close; `min_closes=3` now fires on exactly the 3rd
  consecutive close, matching the documented thesis. Verified numerically and locked with 2 regression
  tests **proven to fail on the old counter and pass on the new** (by injecting the old line).
- **Blast radius — contained (hard evidence).** `capitulation_signals` is NOT wired into any locked
  book: System #2's fear sleeve uses `volatility_premium.fear_signals` (no such counter), and the only
  other references are the module's own research `backtest()`, a `validate.py` scorecard, and a
  `thesis.py` import of the *separate* `adaptive_neutral`. So **System #1/#2 locked numbers are
  unchanged** — this is a research-module correctness fix, not a locked-system change.
- **Evidence the fix mattered + `min_closes=3` confirmed best.** Empirical sweep of the CORRECTED
  signal: on the deep ^GSPC 1990-2026 sample, `min_closes=2` is a LOSER (37 trades, PF 0.90, −0.13%
  exp) while `min_closes=3` is a winner (7 trades, 71% win, +2.9% exp, PF 6.01) and `=4` is too rare
  (1 trade). Because the old counter fired `min_closes=3` on the 2nd close, the buggy default had been
  behaving like the *losing* 2-close trigger — the fix moves the default from a deep-history loser to a
  genuine winner. Default stays **3** (evidence-best = documented thesis). Still thin (7 trades/36yr →
  "rare, high-conviction", not bankable alone), so the prior demotion-for-thin-sample stands; it is no
  longer "noise from a bug", though.

### Audit sweep — 6 parallel Opus agents (verify-with-evidence, then fix only what's real)
An external code audit was run claim-by-claim; each was reproduced before any fix, and false alarms
were left untouched. Full suite **109 → 139 tests** (30 new), all green.
- **Test inversion (ACCURATE → fixed).** The abandoned `prob_up` model had 23 tests; the two
  *trusted* edges had zero. Added `tests/test_mean_reversion.py` (9) and `tests/test_vix_capitulation.py`
  (8) exercising `ibs`/`ibs_signals`/`mean_reversion_signals`/`rsi` and `vix_upper_band`/
  `capitulation_signals`/`adaptive_neutral`.
- **Cache truncation + no TTL (ACCURATE → fixed).** `data/cache.py` keys carried no date range, so a
  cache first filled with a short window was served verbatim for a later WIDER request (repro: asked
  2010-start, silently got 2019-start). Added a coverage check + `max_age` TTL as **keyword-only,
  backward-compatible** opt-ins, plus a sidecar `*.meta.json`; `tests/test_cache.py` (8). Residual
  one-liner: `prices.py` should pass its requested range to opt in (noted in the docstring).
- **Fresh-cache failure (ACCURATE → fixed).** `run_book.py` loads `_VIX`/`_VVIX`/`GLD`, but
  `universe.yaml` declared only VIX and `backfill.py` never warmed the `context:` series → a clean
  checkout raised `FileNotFoundError`. Added VVIX/GLD to `universe.yaml` and a context-warming loop in
  `backfill.py` writing the exact sanitized filenames (`^VIX`→`_VIX.parquet`, …).
- **Dashboard (ACCURATE → fixed conservatively).** Git history confirms pages 1-3 were never built
  (not deleted) and nothing imports the page files; added `pages/README.md` documenting the numbering
  instead of risky renumbering. `7_Self_Learning.py` is synthetic-data only (real `autopsy_trades`
  imported but never called) → added a prominent in-app SYNTHETIC-DATA-DEMO `st.warning`.
- **Hardcoded path (ACCURATE → fixed).** `highconf_eval.py` default `--out` was an absolute
  `/home/user/...` path; now repo-relative via `Path(__file__).resolve().parents[1]` (identical when
  run from root).
- **`cpcv.build_paths` return-type mismatch (ACCURATE → fixed).** Annotation said `list[list[int]]`
  but it returns `list[list[tuple[int,int]]]`; corrected annotation + docstring + a contract
  regression test. (Method is unused but a tested public API — kept, not deleted.)
- **market_internals survivorship (REFINED — half false).** The scary half ("the `dateFirstAdded`
  gate is fake / silently uses today's list") is **INACCURATE**: the add-gate is genuinely enforced
  and effective (proven on cache — effective constituent count rises 389→502 across 2019→2026). The
  real, *already-self-flagged* half is the missing delisted/`dateRemoved` side (no vendor feed on this
  plan) → now an honest runtime `RuntimeWarning` (env-silenceable) + precise docstring instead of an
  overselling one-liner; `tests/test_market_internals_survivorship.py` (4).
- **FALSE ALARMS (verified INACCURATE → no change).** (1) "Double-costing in
  `validate.normalize_ledger`" — cost is charged exactly once in both the `ret` and price branches
  (numeric repro 0.0198 either way). (2) "`backtest/sizing.py:vol_target` is dead code" — it's
  exported public API covered by 13 backtest tests.
- **🔴 New bug found mid-verification (FLAGGED, not auto-fixed).** `vix_capitulation.capitulation_signals`
  fires on the **2nd** consecutive above-band close, not the 3rd, despite the documented "≥3
  consecutive closes" (run-length counter shares the leading non-above bar's group). This changes a
  **locked** book's behavior (System #2 fear sleeve), so it needs its own fix + re-validation cycle
  before changing — not folded into this audit batch.

### Calibrated & locked — System #2: long_swing (4 parallel Opus agents)
- **long_swing (breakout + fear, weeks→~3 months) is locked at max_days = 63** and OOS-validated by a
  4-agent drill (cap-sweep / ledger-integrity / sleeve-decomposition / overfit-firewall):
  - **Cap sweep:** every cap 21-126d is positive in BOTH train and test; tight caps (21/40) demonstrably
    chop live winners (a third of trades hit the clock still winning at cap 21). Runner-harvest plateaus
    by ~63-90d. **63 is the spec-faithful "~3 month" lock** (90 is marginally higher Calmar but the gap is
    inside the noise, and 90 creeps past the stated horizon).
  - **Ledger integrity: CLEAN** — no churn, no same-bar stacking, honest gap fills (all 54 exits inside
    [low,high] or the open). The biggest in-horizon winner (+$96.48/sh) rode the chandelier, not the cap.
    Independently confirmed the 2020 COVID rebound wants to run **458 bars → position-book territory**,
    so capping at 63 is correct horizon discipline, not a clipped runner.
  - **Keep both sleeves:** each clears the firewall standalone; they diversify by holding-TIME coverage
    (fear adds 117 otherwise-flat days, fires across all regimes); combining lifts Sharpe 0.42→0.50,
    Calmar 0.66→0.87. Fear is a thin 19-trade diversifier — sized small.
  - **Firewall verdict — real but FRAGILE:** 27/27 parameter-perturbation cells positive (broad plateau,
    no knife-edge); **NOT a COVID mirage** (drop 2020 → still +66% / Sharpe 1.02; 6 of 8 years positive);
    replicates on SPY/^GSPC/QQQ but **vanishes on small-caps (IWM)**. Deflated Sharpe **0.81–0.86** —
    clears the 0.70 fragile floor, below the 0.95 "REAL" bar. Tradeable as a thin large-cap edge.
  - **Headline (2019-26, SPY, cap 63):** 54 trades, 46% win, **+87.4%**, PF 3.25, maxDD −10.1%,
    in-window Sharpe **1.07**. Train +24.9% / 0.75; Test +50.0% / 1.39 (OOS ≥ train, no decay).

### Fixed (engine — measurement bug found by the firewall drill)
- **In-window Sharpe.** `run_portfolio` computed Sharpe/maxDD/exposure over the FULL price index
  (1993→2026), so the flat pre-window days diluted the annualised Sharpe by ~√(total/in-window) — a
  true **1.07** long_swing Sharpe was reported as **0.50**. Stats are now sliced (and rebased) to the
  study window `[start, end]`. Corrects EVERY book's Sharpe (short_swing 0.38/0.40 → **0.81/0.84**).
  Locked with a regression test (`tests/test_portfolio_engine.py`): leading flat history must not
  change in-window Sharpe/total_return. (107→109 tests pass.)
- **`base`==`guarded` no-op removed.** long_swing/position carry no breadth-gated sleeve, so a "guarded"
  run was byte-identical to base — they now expose a single `base` model; `run_book.py` resolves the
  requested `--model` against what each book actually provides.

### Docs — recommendations captured
- **`ROADMAP.md` now leads with a dated, priority-ordered "next improvements" list** (hardest-evidence
  first): calibrate & lock System #2 (long_swing) and #3 (position); walk-forward the *parameters*
  (cap/threshold/ATR mults) with a multiple-testing-deflated Sharpe; add SPX/QQQ/IWM for sample size;
  emit per-trade MFE/MAE excursion from the engine; re-stream the ensemble on the three CLEAN books;
  honest per-book cost/execution; a regime-sizing overlay. Stale RSI-2 backlog item marked superseded
  by IBS<0.1 (System #1).

### Changed (architecture — THREE horizon-separated books, user-directed)
- **The mixed book is split into three separate systems, one per holding horizon.** Jamming a
  1-10 day dip and a multi-month trend ride into ONE book under ONE exit rule is exactly what
  produced the 134-bar ("WTF") hold inside a "swing" book — the chandelier was riding a trend to
  exhaustion (correct for a long-term swing) but the position lived in a book the desk reads as
  short-term. Fix: `scripts/run_book.py` now selects a `--book` (default `short_swing`), each with
  its OWN sleeve membership and its OWN max-hold clock:
  - **short_swing** — 1-10 trading days. IBS<0.1 oversold dip-buy in an uptrend. Exit: 1-ATR stop,
    3-ATR chandelier, **HARD 10-day cap**.
  - **long_swing** — weeks to ~3 months. Breakout + fear capitulation. Exit: 1-ATR / 3-ATR, 63-day cap.
  - **position** — long hold (months-to-years). 200-SMA trend core. Exit: 1-ATR / 3-ATR, 504-day backstop.
  The 1-ATR-stop / 3-ATR-chandelier rule is identical across books; only the MAX-HOLD differs — that
  is the horizon. "Don't cut home-runners" now means *within a book's horizon*, so a short-swing
  winner is still capped at 10 days while a position winner rides for months.
- **System #1 (short_swing) locked & OOS-validated.** Holds 1-10d only; positive in BOTH train
  (2019-22 incl. bear) and test (2023-26 bull). FULL 2019-26 @ 1.5×: **guarded** 29 trades, 59% win,
  +22.2%, PF 2.61, maxDD **−3.6%**, Sharpe 0.40; **base** 41 trades, 51% win, +26.7%, PF 2.20,
  maxDD −6.3%. The breadth guard wins win-rate / PF / drawdown in every regime (half the DD for a
  modest return give-up). Jan 2025→Jun 2026 slice: guarded NET +$39.84/sh (5 trades), base
  +$40.51/sh (7 trades). long_swing + position books are scaffolded next (not yet calibrated).

### Fixed (ledger layout — LOCKED, user-directed)
- **Every exported ledger now carries the totals block + per-share P&L** — the locked CSV layout
  ("always include WIN $, LOSS $, NET $"). `scripts/run_book.py` gained `_write_ledger()`, which
  appends a 3-row totals block — **TOTAL WIN $ / TOTAL LOSS $ / NET TOTAL $** — summing a new
  `pnl_per_share` column (per-share P&L = exit − entry − entry×0.0002, the ~2bp SPY round-trip cost).
  `pnl_per_share` sits right after `ret`; WIN/SCRATCH dollars are summed to the win side, LOSS to the
  loss side (a +1-ATR/breakeven exit is a WIN/SCRATCH, never a LOSS). Both per-model exports and the
  combined `_all.csv` route through it. Re-baseline FULL 2019-26 @ 1.5×:
  **base** 82 trades, NET **+$618.21/sh** (WIN +$862.04 / LOSS −$243.83);
  **guarded** 70 trades, NET **+$615.54/sh** (WIN +$810.95 / LOSS −$195.41).

### Decode campaign — Round 1 (5 parallel Opus agents)
- **TWO real new edges cleared the firewall:** (1) **Reaction-PEAD** — buy the earnings-reaction
  gap-up, hold 10-20d: OOS +9-13%/yr, t≈2.1, broad (~450 names); (2) **cross-sectional ML ranker**
  (3-4wk) — market-neutral, net Sharpe 0.79, alpha ~9%, **PBO 0.029, shuffle-canary holds**. Plus a
  **validated, persistent regime model** (stress regimes = highest fwd returns; re-confirms
  capitulation-long). See DECODE.md.
- **Mirages killed:** naive cross-sectional factors (survivorship + crash-beta), cross-asset TSMOM
  (sample-starved, worse-than-random on 2010+), analyst grades, insider (leakage trap), congress.
- **Deep verdict:** ~38% of the market is systematic/decodable (market factor + regime), ~62%
  irreducible at retail cost. Tradeable decode = regime + catalyst + thin ML sleeve, NOT stock-picking.
- **Data repaired:** deep history pulled to inception (indices/VIX/yields→1990, sectors→1998,
  bonds→2002) so trend/regime edges test across dot-com + GFC. Survivorship-free universe in progress.

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
