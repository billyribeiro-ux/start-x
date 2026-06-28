# DECODE — the running ledger of the market-decode campaign

The firewall (out-of-sample, deflated Sharpe, PBO, leakage/shuffle canary) decides what is real. A
"decode" that doesn't clear it is a mirage and is logged as such. Newest round on top.

---

## SHORT-SCANNER campaign — 4 parallel angles (cross-section / breakdown / event-drift / regime) → VERDICT: NO standalone short alpha

**The question:** the index is un-shortable (upward drift). Does a real, firewall-cleared SHORT edge exist in the
**single-name universe** (~1,186 names incl. delisted)? Four agents attacked it in parallel; I reproduced every
verdict from their artifacts. **All four fail the firewall — and fail for the SAME structural reason: weak /
breaking-down / gapping-down names BOUNCE** (the oversold mean-reversion that powers our *long* IBS edge is exactly
what runs over a short). Net of ~2bp + realistic borrow. TRAIN 2018-22 / TEST 2023→2026-06-25.

| Angle | Method | OOS (2023→now) short Sharpe, net | DSR | PBO | Gate | Cause of death |
|---|---|---|---|---|---|---|
| **A — cross-sectional short leg** | short worst-decile by momentum/RS/trend, 28 configs | **−0.78** (short-only); mkt-neutral +0.35 | 0.000 / 0.082 | 0.67 / 0.00 | ❌ | weakest decile *outperforms*; short leg is pure −1.3 beta. Borrow break-even at **−18.7%/yr** (hopeless). Mkt-neutral "+" is the LONG leg. |
| **B — single-name breakdown** | breakdown/failed-rally/rel-weak/lower-high, 10 variants, 312k trades | **−1.7 to −6.1** (every variant) | 0.00 | — | ❌ | ~30% win — short the break, it bounces, 1-ATR stop runs over. Best variant +0.26% in TRAIN → −0.55% TEST (overfit). |
| **C — negative event-drift** | down-gap ≤−5% on ≥1.5× vol, weak close; mirror of Reaction-PEAD | drift fwd5 −0.32% → fwd20 **+1.10%** (INVERTS) | — | — | ❌ | down-gaps *recover*; no downside continuation. Tiny 5d dip < cost. Opposite of a PEAD-short signature. |
| **D — regime gate** | short worst-decile conditional on 18 bear/breadth/vol/credit flags | **negative in EVERY flag**, TRAIN and TEST | — | — | ❌ | NO regime (below-200, death-cross, weak-breadth, vol-stress) flips shorting positive. Even deep-bear gates lose. |
| **E — overbought MIRROR (endpoint)** | short the OVERBOUGHT (IBS≥.9, RSI2≥95/98, ext20/50, BB-upper, blow-off gap), 8 setups × 3 trends, 1.89M signals, fixed-horizon hold | fwd ret **POSITIVE everywhere** | — | — | ❌ | held to the endpoint, over-extended names MOMENTUM up; blow-off gap-ups in a downtrend forward **+20%** (squeezes). |
| **F — pullback SCALP (path)** | short over-extension (≥10% > 20-SMA), COVER FAST on a 1-ATR retrace, 1-ATR stop, 3-day cap; weak-tape gate (SPY<20-SMA) | **+0.40%/trade OOS on full universe → KILLED: −0.17%/trade on LIQUID names** | 0.49 (liquid) | — | ❌ | looked positive on the FULL universe, but pre-registered walk-forward exposed it as a MIRAGE — microcap/bad-data outliers (trades −267%..+98%) + a breadth illusion. On tradeable names (>$5, >$20M ADV) it is negative, Sharpe ~0. |

**The keystone insight — and the trap the user's push surfaced (then the firewall caught):** as an *endpoint* bet,
single-name equities are **long-biased at both tails** — oversold revert UP (powers the long IBS edge, kills
weakness-shorts A/B/C/D), overbought continue UP (kills the held-to-endpoint mirror E). The **path** angle (F) was
the one that looked alive: a fast SCALP of the intraday retrace, gated to a weak tape, showed +0.40%/trade OOS and a
pre-registered walk-forward Sharpe ~1.9 / DSR 1.0. **That was TOO GOOD — and it was a mirage.** Adversarial
verification killed it: the per-trade returns ran −267%..+98% (microcap / bad-data / delisted-to-zero junk you can
never short), the median trade was +0.01% (zero), the high Sharpe came from a *breadth illusion* (averaging thousands
of correlated names crushes the daily vol), and it wasn't even beta (corr to short-SPY ~0). **Filtered to liquid,
tradeable names (>$5, >$20M/day ADV) the edge is NEGATIVE (−0.17%/trade, Sharpe −0.01, DSR 0.49).** The "retrace to
the mean" is descriptively real but too small to trade net of costs on liquid names. Lesson reinforced: a
spectacular daily-book Sharpe across a huge junk universe is a red flag, not a trophy. Repro: `scripts/short_pullback_scalp.py`
(config search), `scripts/short_pullback_regime.py` (weak-tape gate, full universe), `scripts/short_pullback_walkforward.py`
(the kill — FULL vs LIQUID + the short-SPY beta benchmark).

### "Both ways" / regime-conditional check — does shorting work in a confirmed BEAR? (mostly no)
The point of decoding shorts is a both-ways ML for when the bull ends. So we tested the fair question:
short the INDEX gated to a confirmed bear (SPY < 200-SMA), 2007-2026 (cache covers the 2008/2020/2022
bears). `scripts/short_regime_overlay.py`:
- **Naive daily-hold short, bear-days only:** ann **−15.9%**, Sharpe −0.30. Shorting loses even *in* the
  bear — because bear markets are **rally-infested** (the sharpest up-days happen in downtrends) and the
  200-SMA is laggy. Only **2008 (+28.8%)** paid; 2020 −18.6%, 2022 −4.5%.
- **Trailing TREND short (short fresh lows in a bear, ride a 3-ATR chandelier):** still loses overall
  (Sharpe −0.23, −47% total). It rescues 2009/2020 but the chandelier whipsaws out of 2008's grind
  (+28.8% → +3.8%) and 2022's rallies chop it to −18.1%.
- **Verdict:** NO directional index short — blended, regime-gated, or trend-ridden — is profitable across
  20 years incl. three bears. **The market is structurally asymmetric: "both ways" is NOT symmetric.**
- **So what IS the both-ways mechanism?** (a) **Regime-switch long→CASH** in bears = System #3 position
  book (validated: ~half the drawdown), the legitimate defensive both-ways — flat, not short; (b) the
  **market-neutral ranker** where the short leg is RELATIVE (long strong / short weak), financed and
  hedged, never directional; (c) a tiny −beta tail hedge. The ML's job is **regime detection + relative
  ranking**, NOT learning a directional short signal (there isn't one, even for the bear).

**The decoded answer (firewall-grade):** *no tradeable single-name SHORT edge exists in this universe.* All six angles
(A/B/C/D/E endpoint + F path) are now dead on tradeable names. F survived longest — the pre-registered walk-forward
was run exactly as planned to confirm-or-kill it, and it **killed it** (liquid-only Sharpe ~0; the "+0.40%" was junk
outliers + a breadth illusion). The short side ships only in two proven roles: (1) the **short leg of a market-neutral
book** (the ML ranker, Sharpe ~0.87 — the long leg finances it), and (2) a **small −beta drawdown HEDGE**
(`overbought_short`, sized small, not alpha). **The qedge scanner correctly returns "LIKELY OVERFIT" for single-name
shorts — that is the firewall working, not a bug.** A standalone short scanner is the wrong instrument on this
universe; the right one is the market-neutral ranker's short leg. The honest scanner deliverable is a *candidate
surfacer* that lists names triggering the (research-only) short setup with their true firewall status — never a
"profitable short signal," because none survives liquidity + cost + deflation. Scripts: `scripts/short_xsec.py` (A), `scripts/short_regime_gate.py` (D);
B/C harnesses in the decode scratchpad; `scripts/short_overbought_study.py` (E, the mirror) and
`scripts/short_scan_study.py` (B+C reproducers). n_trials honestly counted (A=28, B=10, D=18, E=24) so DSR isn't flattered.

---

## CONSOLIDATED SCOREBOARD (all edges characterized, gathered before assembly)

> 🏁 **THE DECODED SYSTEM — the honest final answer (2026-06-26, after the survivorship + investability audits).**
> The market's *bankable* decodable structure is ONE engine: the **better long model** — vol-targeted
> risk-parity across the two regime-aware swing books (short-term IBS mean-reversion + long-term breakout/fear
> capitulation). **Sharpe 1.22, CAGR 13.3%, maxDD −14.8% (2018-26)** — beats SPY (0.72 / −34% DD) on risk-adjusted
> return and drawdown. Everything else either died or doesn't add:
> - **ML cross-sectional ranker** → survivorship MIRAGE (−0.18 OOS survivorship-free). Struck.
> - **All shorts** (5 angles + regime) → no tradeable edge; market is long-biased at both tails. Struck.
> - **Stress-gated SWING ALPHA** (the scanner) → a REAL per-trade pattern (+0.90%/trade, alpha +0.92%, beta 0.03,
>   generalizes ETFs/indexes/stocks) BUT as an investable capped book it's Sharpe 0.84 / **−36% DD**, and adding it
>   to the long model DRAGS the ensemble to 0.82 (−0.40 Sharpe) despite +0.05 correlation. Per-trade alpha ≠
>   investable book. NOT a portfolio improver — kept as a research/scanner pattern, not a sleeve.
> - **Position book** (200-SMA) → drawdown-defense overlay (its own benchmarked mandate), not an alpha adder.
> The earlier "constellation of edges assembled into ~1.5 Sharpe" was the survivorship/investability illusion.
> The honest decode: **the long mean-reversion+breakout+fear engine is the edge; the rest is noise, beta, or
> un-investable patterns.** Repro: `scripts/decode_ensemble.py`, `scripts/long_model.py`.
>
> 🔁 **SELF-LEARNING UPDATE (R5, 2026-06-28) — "can we learn from the losers and avoid them?" → NO, proven.**
> A new self-learning loss loop (`scanner/selflearn.py`, `scripts/swing_selflearn_loss.py`) dissects every
> losing swing trade, distills pre-registered AVOID rules + a walk-forward 2nd-stage P(loss) veto, and judges
> them on a purged out-of-sample firewall. It ran to convergence (6 trials, all rejected): **the loss-prone
> cohort has the FATTEST RIGHT TAIL** — every veto removed *net-profitable* trades (vetoed cohorts paid
> +2.0% to +3.7%/trade). The variance IS the edge; the losses are the irreducible cost of the bet, not a
> separable subset. The WHY is documented (losers are shallower dips — lower rsi2/range — that bleed through
> the 1-ATR stop; worst trigger = pullback 57%). The self-learning layer is wired into the live scanner but
> correctly holds **0 avoid rules** — and that null result is the deliverable: it blocks a plausible-but-wrong
> filter that looked great in-sample (higher loss *rate*) but cut returns OOS. See LEDGER R5 + SELFLEARN_LOG.md.

**The shape of the answer (original Round-1 framing, now superseded by the box above): a constellation of
small weakly-correlated edges — but the survivorship + investability audits collapsed most of them.**

> ⛔ **CORRECTION (2026-06-26) — the "ML cross-sectional ranker" was a SURVIVORSHIP MIRAGE.** When rebuilt on a
> point-in-time, survivorship-FREE universe (`startx.data.membership`, incl. delisted names), the identical pipeline
> on the identical 2018-2026 window gives **OOS Sharpe −0.18 (DSR 0.01)** — and reproduces the old **+0.83 (DSR 0.77)
> ONLY when fed today's current-survivors** (the biased universe the research used). The whole "edge" was buying
> tomorrow's known survivors in the past. The naive 12-1 momentum baseline shows the same collapse (+0.40 biased →
> +0.08 honest). Per staged-search the kill condition is met → **the ranker is NOT a real edge and is struck from the
> ensemble.** Repro: `scripts/xsec_ranker.py [--survivors-only]`. This invalidates the "best diversifier" and the
> assembled-ensemble Sharpe below (which weighted it 26%). The validated system is the rule-based books, not this.

### ✅ REAL, tradeable edges (each thin alone — DSR mostly 0.2–0.6 — but firewall-cleared)
| Edge | OOS evidence | corr to equity book | role |
|---|---|---|---|
| **SPY swing book** (breakout/IBS/fear) | un-levered Sharpe ~1.18, maxDD −9% | 1.0 (it IS the book) | core |
| ~~**ML cross-sectional ranker** (4wk)~~ | ~~net Sharpe 0.87~~ → **−0.18 survivorship-free (MIRAGE, struck)** | ~0 | ❌ removed |
| **HMM regime-boundary capitulation** | fwd5d +0.85%, hit 72.6%, **t 3.14** (beats VIX band t 0.94) | overlaps fear sleeve | upgrade the fear trigger |
| **Reaction-PEAD** (earnings gap-up drift) | +9–13%/yr, t 2.1, DSR 0.14–0.27, sector-neutral survives | +0.08 | thin, orthogonal |
| **Index-inclusion drift** | +1.27% post-2010, t 3.2, DSR 0.34, ~20 events/yr | ~0 | tiny, orthogonal |
| **Macro PPI-surprise → equities** | t up to −5.16 but DSR 0.25 after deflation | ~0 | overlay only, not primary |
| **Stat-arb residual basket** | net Sharpe 0.57, PBO 0.26, canary p 0.02 | overlaps IBS (it's long-the-loser) | confirms mean-reversion |

### ❌ MIRAGES killed by the firewall
naive cross-sectional factors (survivorship + crash-beta) · cross-asset trend / managed-futures
TSMOM (**dead even on deep 1998-2026 history** — the sample-starved hope failed) · credit-gated
rotation (inverts on deep history) · classic stat-arb pairs (distance & cointegration — zero gross
edge, post-2003 collapse confirmed) · index deletion-reversal (a GFC-outlier mirage) · analyst
grades (decay OOS) · insider buying (leakage trap) · congress flow · most macro releases (post-2020
regime artifacts).

### 🧮 The deep verdict — how decodable is the market?
**~38% systematic, ~62% irreducible at retail cost.** Direction & regime are decodable and
persistent; single-name residual alpha is mostly noise once you pay to trade it. The decoded,
tradeable structure = **regime + a market-neutral ML sleeve + a few thin event/catalyst edges** —
NOT stock-picking, NOT one magic signal.

### 🔭 The ensemble math (the path, quantified — assembly pending all results)
Max-Sharpe (not equal-risk) weighting; each near-independent edge raises the ceiling √(ΣSRᵢ²):
SPY book (1.18) **+ ML ranker (0.87, β≈0) → ceiling ~1.45**; + the orthogonal thin edges (PEAD,
index, macro) nudges toward ~1.5. Realistic decoded-system target: **a ~1.45–1.55 Sharpe diversified
ensemble** — world-class *because* it's a portfolio of real edges, not a fragile single signal.

### 🏁 ASSEMBLED — the decoded system (real streams, monthly, max-Sharpe; `portfolio/ensemble.py`)
> ⛔ **VOID (2026-06-26):** this ensemble weighted the ML ranker 26%, and the ranker is a survivorship mirage
> (−0.18 OOS survivorship-free, see correction above). The 1.38 / DSR 0.995 headline does not stand. The real
> system is the rule-based books + the better long model; a re-assembled ensemble must use only survivorship-free,
> firewall-cleared streams. Re-build pending.

| Edge | Sharpe | weight | corr to book |
|---|---|---|---|
| SPY swing book | 1.12 | 72% | 1.00 |
| ~~ML ranker (market-neutral)~~ | ~~0.90~~ **mirage** | ~~26%~~ → 0% | — |
| Reaction-PEAD | 0.44 | 2% | 0.23 |

**Ensemble: Sharpe 1.38 · deflated Sharpe 0.995 · CAGR +12.9% · maxDD −8.7% · Calmar 1.49** (89 months).
[DSR corrected 2026-06-21: the prior "0.61" came from a since-fixed *pessimistic* bug in
`ensemble._deflated_sharpe` (double `/√ppy`); the audited Bailey/LdP formula gives **0.995** on the
same 89-month stream — the ensemble clears the 0.95 "REAL" bar.]
vs SPY buy-and-hold (0.85 / +15.8% / −34.1% / 0.46): ~the index's return at a quarter of the
drawdown, 62% higher Sharpe, near-zero correlation. The ML ranker lifts the book +0.25 Sharpe at
corr 0.09 — real diversification. The thin orthogonal edges (index-flow, macro-PPI, stat-arb) and the
HMM regime trigger would each add a little more (not yet streamed in). Output: `decoded_ensemble_monthly.csv`.


---

## Round 1 — five parallel frontiers (cross-section / catalysts / factor-regime / ML / cross-asset)

### ✅ REAL edges (cleared the firewall)
1. **Reaction-PEAD** (catalyst, single stock). Buy the stock that GAPPED UP on its earnings-reaction
   day; enter next session; hold 10–20d. OOS 2019-26 calendar-time net **+9–13%/yr, t≈+2.1, Sharpe
   ~0.8**, positive in BOTH train and test, drift monotonic 5→10→20d (the PEAD signature), broad
   (~450 names). Caveat: deflated Sharpe ~0.05–0.08, top-5 names ≈ half the PnL → promising, not yet
   bankable alone. *Long-only, event-driven — diversifies the SPY swing book.*
2. **Cross-sectional ML ranker** (3–4 week horizon). GBM over price/liquidity features, market-
   neutral. OOS net **Sharpe 0.79, alpha ~+9%/yr at beta≈0, PBO 0.029, shuffle-canary holds**, DSR
   0.60. Features: 1m momentum + Amihud illiquidity + reversal + trend. Caveat: DSR <0.95 (size
   modestly), −29% momentum-crash DD needs an overlay, survivorship flatters the long leg.
3. **Regime model** (structural). K-means (K=4) on a 7-dim market-state vector is **persistent OOS**
   (88.6% stay-prob, ~8.8d runs, train≈test). Stress/capitulation regimes have the **highest forward
   returns** — a 4th independent re-confirmation that high-VIX is a capitulation LONG. Doesn't
   manufacture Sharpe (all regimes forward-positive) but is a real decode → use for conditioning/sizing.

### ❌ MIRAGES killed by the firewall
- **Naive cross-sectional factors** (momentum/reversal/low-vol/size/value): dead/negative OOS;
  "alpha" is short-beta crash-hedge + survivorship inflation (high-vol survivors invert low-vol).
- **Cross-asset / TAA / managed-futures TSMOM**: no significant alpha on the 2010+ sample (TSMOM
  worse than random, canary p=0.60) — but **sample-starved**; real diversification value, and
  risk-on/off (trend + credit gate) is the one live thread (t=1.44).
- **Analyst grades** (decay OOS), **insider buying** (survivorship/leakage trap, negative OOS),
  **congress flow** (noise), **single-name residual reversal** (real gross SR 0.57, dies on cost).

### 🧮 The deep verdict — how decodable is the market?
**~38% systematic, ~62% irreducible at retail cost.** PC1 (the market) ≈ 38–40% of cross-sectional
variance; after it a long flat tail (no clean factor zoo). The market's *direction & regime* are
decodable and persistent; single-name *residual alpha* is essentially noise once you pay to trade it.
The tradeable decode is **regime + catalyst + a thin ML cross-sectional sleeve — not stock-picking.**

### 🔧 Data fixes (the binding constraint, now being removed)
- **Deep history** pulled to inception (^GSPC/^VIX/yields→1990, SPY→1993, sectors→1998, bonds→2002):
  the cross-asset/trend/regime edges can finally be tested across the dot-com crash AND the GFC, not
  one bull. (DONE.)
- **Survivorship-free universe** (FMP historical constituents incl. delisted names): needed by the
  cross-section + ML threads (current 503 inflates the long leg). (IN PROGRESS for Round 2.)

### → Round 2 (launched): build the survivors, on the repaired data
A) **Reaction-PEAD → productionize + enrich** (chandelier exit for the right tail, stack surprise +
   analyst PT-raise + volume, sector-neutral, regime-condition).
B) **ML ranker → harden** (H=4w default, add FMP fundamentals, LGBMRanker + ensemble to lift DSR,
   momentum-crash risk overlay) on the survivorship-free universe.
C) **Cross-asset / trend / regime on DEEP history** + a true **HMM** with transition probabilities
   (enter *before* capitulation forms) + regime-conditional capitulation sizing.
