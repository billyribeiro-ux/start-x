# TRIALS LEDGER — every configuration tested, with N fed to the deflation bar

Per the Operating Standard (v2): breadth is only safe if every trial is counted. `N` is the running
count of distinct configurations searched; it feeds the Deflated Sharpe / PBO bar so the threshold to
declare an edge rises automatically as we explore. **A Sharpe with no N attached is not a result.**
Real-data window is **2018-01-01 → 2026-06-26** (FMP/cache coverage is ~96-100% here; pre-2016 is
survivorship-contaminated and its numbers are void — see the membership coverage report). Newest first.

Hypothesis gate (must be answerable before a method is tried): (1) economic rationale, (2) who is on
the other side and why they keep losing, (3) the pre-stated kill condition.

---

## Round R5 — self-learning LOSS loop (2026-06-28)  ·  VERDICT: the losses are NOT separably avoidable (firewall held)

**Hypothesis:** the ~52% of stress-gated swing takes that lose share an ex-ante signature a model can learn
to AVOID, lifting the book's risk-adjusted return. **Other side:** if true, we'd be the only ones leaving
that money on the table — unlikely for a public-feature mean-reversion edge. **Kill (pre-stated):** if no
avoid mechanism lifts expectancy AND PF AND DSR out-of-sample with the vetoed cohort genuinely worse, the
losses are the irreducible cost of the bet — do not filter. Engine: `scanner/selflearn.py` +
`scripts/swing_selflearn_loss.py` (purged walk-forward over the full 366-trade taken book, ETFs + 120
survivorship-free liquid stocks; `scanner_memory.json` persists; converges at 3 non-promotes).

**The WHY (post-mortem, descriptive):** loss rate by trigger `pullback` 57% > `mean_rev` 53% > `rs_break`
45% > `squeeze` 21%; by regime risk_off 54% ≈ crisis 49%. Losers are SHALLOWER dips (lower rsi2, lower
pos_range20/bb_pos — they buy a not-deep-enough pullback that keeps bleeding through the 1-ATR stop).

**Every avoidance mechanism tested REMOVED net-profitable trades (the decisive result):**
| # | avoid mechanism | OOS (walk-forward) result | vetoed cohort | verdict |
|---|---|---|---|---|
| R5.1 | ML 2nd-stage P(loss) veto, q=0.80 | exp +1.21%→+0.88%, DSR 0.85→0.44 | **+2.79%** | reject |
| R5.2 | ML P(loss) veto, q=0.85 | exp +1.21%→+1.04%, DSR 0.85→0.61 | **+2.22%** | reject |
| R5.3 | ML P(loss) veto, q=0.90 | exp +1.21%→+0.98%, DSR 0.85→0.60 | **+3.71%** | reject |
| R5.4 | rule: AVOID rsi2 low tail (shallow dip) | exp +1.21%→+0.93% | +2.02% | reject |
| R5.5 | rule: AVOID trigger=pullback | exp +1.21%→+1.23% (PF/DSR down) | +1.13% | reject |
| R5.6 | rule: AVOID dist_sma200 high tail | exp +1.21%→+1.06% (vetoed BETTER) | +1.79% | reject |

**Finding:** the cohort the loss-model flags as "most likely to lose" is the cohort with the FATTEST RIGHT
TAIL — the deepest-stress entries lose more often but pay the most when they work. Trimming the left tail
trims more of the right. **The variance IS the edge.** Self-learning loss-avoidance CONVERGED to the empty
set (0 promoted rules) — and that null result is the value: it blocks a plausible-but-wrong avoid filter
that would have quietly cut returns (it looked great in-sample: vetoed cohorts have higher loss *rates*).
The raw stress-gated book stands unfiltered: **n=366, exp +0.95%/trade, PF 1.60, DSR 0.91.** The
self-learning layer is now wired into the live scanner (it will flag-and-explain any avoid rule that ever
DOES clear the firewall as new data arrives) but currently holds nothing — correctly. **+6 trials → N.**

---

## Round R4 — swing meta-label harness (2026-06-26)  ·  VERDICT: harness SOUND; signal thin/regime-conditional

**Hypothesis:** regime×location×trigger swing setups on tradeable ETFs have a take/skip edge a meta-model
can size. **Other side:** liquidity providers / late momentum chasers. **Kill:** if no threshold clears
DSR>0.5 AND PBO<0.5 cost-adjusted OOS, don't promote.

P4 harness PASSES its leakage canary (the gate that makes everything else trustworthy):
- shuffled-label OOS AUC **0.510** (~0.5 = noise has no edge) ; lookahead feature AUC **1.000** (leak caught).

Walk-forward meta-model (10 ETFs, 7,238 events 2012-26, next-open fills, 3bp, N=15):
| thr | n | exp/trade | PF | DSR | gate |
|---|---|---|---|---|---|
| 0.45 | 864 | +0.21% | 1.28 | 0.92 | — |
| 0.70 | 96 | +0.49% | 1.72 | 0.75 | — |
PBO (CSCV, 11 thresholds) = **0.55** (> 0.5) -> threshold selection marginally overfit -> **does NOT promote**.
Regime-conditional: edge concentrates in stress — risk_on +0.16% / risk_off +0.23% / **crisis +0.77% PF 1.87**
(n=39). Consistent with every capitulation finding. Thin & regime-dependent; not a clean firewall pass.
Scripts: scripts/swing_validate.py.

**P5 self-learning (scripts/swing_learn.py, N=16 incl. the pre-registered regime gate):**
- REGIME-GATED (stress = risk_off+crisis, theory-motivated by the capitulation prior): exp **+0.68%/trade,
  PF 1.82, Sharpe 1.35, DSR 1.00** — the strongest, most real signal in the project (3x the pooled book).
  BUT **PBO 0.76** on threshold selection -> threshold choice is unreliable; pre-commit a threshold (no
  selection) and it's a legitimate regime-conditional edge, pending a touch-once forward holdout. Does NOT
  yet get the formal "clears" stamp (PBO>0.5 with selection).
- OOS MDA feature importance (promote/decay): **regime_stress #1** (+0.019), then dist_sma200, bb_pos,
  ret21, atr_pct, rs_spy20. RETIRE (no OOS skill): nr7, rsi_div, vol_thrust, rsi2, rsi14, dist_avwap,
  dist_plow20 — the meta-model's skill is regime + location + medium momentum, NOT short-term oscillators.
- Drift: strong in volatile years (2020/2023-25 +1.5..2.2%), weak in calm grinds (2017/18). Regime-dependent.
- VERDICT: a real, economically strong, regime-conditional swing edge — the best found — but threshold-
  selection-fragile; honest status = promising-not-yet-stamped. Next: pre-commit threshold + touch-once holdout.

**P5 touch-once HOLDOUT (scripts/swing_holdout.py) — ✅ FIRST PROMOTED SIGNAL.** Removing the threshold
SEARCH (the cause of PBO 0.76) and pre-registering the rule fixed it. DEV 2012-19 (eyeball) / HOLDOUT
2020-26 (touched once, no search):
| rule | HOLDOUT n | exp/trade | PF | Sharpe | DSR(N=2) |
|---|---|---|---|---|---|
| stress raw (all events) | 347 | +0.41% | 1.37 | 0.68 | 0.98 |
| **stress + prob>=0.5** | 88 | **+0.68%** | 1.71 | 1.17 | 0.96 |
Skeptic checks on the promoted rule: DSR vs N = 0.96(2)/0.78(8)/0.66(16)/0.55(30)/0.46(50) — survives
honest deflation to N~30, THIN beyond. **BETA CHECK (decisive): SPY fwd-10d on the same entries = −0.35%
(tape falling) while the strategy makes +0.68% -> beta 0.12, ALPHA +0.73%/trade.** Real timing/selection
alpha, NOT bounce-beta. Charter promotion gate (DSR>0, no-search-so-PBO-n/a, OOS exp>0, regime-conditional
+ alpha shown) -> **PROMOTE small + monitor decay.** First genuinely promotable signal in the project;
honest status = real, regime-conditional swing alpha, THIN — size accordingly. (Caveat: 2020-26 was partly
observed in prior exploration, so the conservative-N DSR ~0.55-0.66 is the fair read, not the N=2 0.96.)

**P6 scanner output + P7 horizon generalization (scripts/swing_scan_live.py, swing_horizon.py):**
- P6: live scanner surfaces stress-gated setups with CALIBRATED conviction (isotonic, honest ~33% = base
  hit rate, +EV on 2:1 payoff), ranked SHAP driver attribution (regime_stress dominant), 1-ATR
  invalidation, regime-matched exp/CVaR. Gate=raw prob (validated rule), display=calibrated. No bare scores.
- P7 horizon: the edge is SWING-HORIZON-SPECIFIC. Touch-once holdout, stress+prob>=0.5:
  SHORT_SWING 3d -> exp +0.10%, PF 1.11, DSR 0.07, **alpha -0.13% (beta 0.31)** — alpha gone, mostly beta.
  SWING 10d -> exp +0.68%, PF 1.71, DSR 0.66, alpha +0.73% (beta 0.12). The selection-alpha needs the full
  ~10d recovery window; compressing to 3d kills it. v1 swing horizon confirmed; don't shorten it.
- EOD-feasible charter P1-P7 COMPLETE. DAY/0DTE/SCALP + gamma/charm/DIX/net-liquidity = data seams (need
  intraday + OPRA + FRED), out of scope on this plan. Promoted signal stands: stress-gated 10d swing alpha,
  thin (size small, monitor decay).

**Hardening — SINGLE-NAME generalization (scripts/swing_singlenames.py): the edge covers ETFs + indexes +
STOCKS.** Survivorship-free PIT universe, 200 liquid names, 76,065 events, touch-once holdout 2020-26:
| rule | n | exp | PF | alpha | beta |
|---|---|---|---|---|---|
| stress raw (all) | 7134 | +0.55% | 1.33 | +0.43% | 0.21 |
| stress + prob>=.5 | 512 | **+0.90%** | 1.63 | **+0.92%** | **0.03** |
STRONGER on stocks than ETFs (+0.90 vs +0.68/trade), near-PURE alpha (beta 0.03), big breadth. Sharpe is
construction-dependent and bracketed: per-trade ~0.9 (conservative) to a naive calendar-book ~3 (OVER-
diversified — averages 100s of correlated stress-day names as if independent; NOT a reliable investable
number, the same illusion as the short breadth case). The robust, construction-free facts are the
expectancy (+0.9%) and ALPHA (+0.92%, beta 0.03). VERDICT: the stress-gated swing edge generalizes across
ETFs/indexes/stocks and is breadth-scalable — real, near-pure alpha, thin->size small. The scanner's three
pillars are validated.

**DECODED ENSEMBLE re-assembly (scripts/decode_ensemble.py) — the swing alpha does NOT improve the system.**
Real edges only, 2018-26, swing alpha built as an INVESTABLE capped book (<=10 concurrent, no over-
diversification illusion):
| stream | Sharpe | CAGR | maxDD | Calmar |
|---|---|---|---|---|
| better long model | **1.22** | 13.3% | -14.8% | 0.90 |
| swing alpha (capped) | 0.84 | 10.1% | **-36.0%** | 0.28 |
| decoded ensemble (RP combine) | 0.82 | 9.6% | -20.5% | 0.47 |
| SPY buy&hold | 0.72 | 12.7% | -34.1% | 0.37 |
corr(long_model, swing_alpha)=+0.05 (low) BUT adding the swing alpha DRAGS the ensemble to 0.82 (-0.40
Sharpe). **Per-trade alpha != investable book**: the swing edge's +0.90%/trade is real, but as a realistic
capped book it's Sharpe 0.84 / -36% DD (a capitulation-bounce that gets run over before the bounce,
concentrated in stress). Low correlation can't rescue a weaker standalone risk profile. VERDICT: the
bankable decoded CORE is the better long model (Sharpe 1.22); the swing alpha is a real PATTERN, not a
portfolio improver. Live 3-pillar scanner (ETFs/indexes/stocks) confirmed working — gate correctly OFF
in the current neutral regime.

## Round R3 — cross-sectional ML ranker (2026-06-26)  ·  VERDICT: MIRAGE (survivorship)

**Hypothesis:** weak cross-sectional predictability (momentum/reversal/illiquidity) lets a market-neutral
long/short ranker harvest relative mispricing. **Other side:** index-constrained & flow-driven holders.
**Kill condition (pre-stated):** if the naive 12-1 momentum baseline shows no OOS cost-adjusted signal on
the survivorship-FREE universe, stop.

| # | config | universe | OOS 2018-26 net Sharpe | DSR | beta | result |
|---|---|---|---|---|---|---|
| R3.1 | LGBM reg, 10 px/liquidity feats, 21d, monthly, decile, 5bp+5%borrow | **survivorship-FREE (PIT)** | **−0.18** | 0.01 | +0.00 | dead |
| R3.2 | naive 12-1 momentum, same construction | survivorship-FREE (PIT) | +0.08 | 0.07 | −0.20 | dead (baseline) |
| R3.3 | R3.1 pipeline, **current-survivors universe** (A/B control) | biased | +0.83 | 0.77 | −0.01 | the bias, quantified |
| R3.4 | naive momentum, current-survivors (A/B control) | biased | +0.40 | 0.29 | −0.16 | the bias, quantified |

**Finding:** the entire ranker "edge" (DECODE's claimed 0.87) is survivorship — it reappears at +0.83 on
current-survivors and is −0.18 on the honest PIT universe. Kill condition met (R3.2 baseline dead). The
ranker is struck from the system. Foundation that caught it: `startx.data.membership` (PIT, incl. delisted).

## Round R2 — directional / regime SHORT (2026-06-26)  ·  VERDICT: no tradeable short edge
Short pullback-scalp & regime overlay. Real-data, liquid names only, net of cost+borrow.
| # | config | OOS result | result |
|---|---|---|---|
| R2.1 | weak-tape pullback-scalp, full universe | "+0.40%/trade" | MIRAGE — microcap/junk outliers + breadth illusion |
| R2.2 | same, **liquid only (>$5,>$20M ADV)** | −0.17%/trade, Sharpe −0.01, DSR 0.49 | dead |
| R2.3 | regime index short (bear-gated, naive + trend-ride) | −47% to −70%, only 2008 paid | dead (bears are rally-infested) |

## Round R1 — the short-decode campaign A–F (2026-06-26)  ·  VERDICT: no standalone short alpha
5 angles + mirror; n_trials counted per angle (A=28, B=10, D=18, E=24, F=256). All firewall-rejected on
liquid names. Detail in DECODE.md (single-name equities are long-biased at both tails). **Sub-total N ≈ 336.**

---

## Running project N (fed to every DSR/PBO from here)
Short campaign ≈ 336 · ranker A/B = 4 · misc baselines ≈ 10 · R5 loss-avoidance = 6 → **N ≈ 356** and counting. With N this large,
the deflation bar is high by design: only edges with a strong prior + clean OOS + low search cost clear it.
Validated production systems (the three rule-based books, the better long model) were each established under
their own pre-registered, low-N protocols — see CHANGELOG/STRATEGY/FINDINGS — and are NOT diluted by this
exploratory N. New exploratory trials add to N here; pre-registered confirmations are logged separately.
