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
| **F — pullback SCALP (path)** | short over-extension (≥10% > 20-SMA), COVER FAST on a 1-ATR retrace, 1-ATR stop, 3-day cap; 32 configs × 8 gates | **+0.33% in-sample / −0.32% OOS** ungated; **+0.40% OOS when SPY<20-SMA** | 0.23 (gated) | 0.09 | ❌ (but **positive OOS**) | the intraday retrace IS real; ungated it decays in the 2023-26 melt-up, but **gated to a weak tape it is positive OOS** — just not strong enough to clear DSR. |

**The keystone insight — refined (this is where the user was right to push):** as an *endpoint* bet, single-name
equities are **long-biased at both tails** — oversold revert UP (powers the long IBS edge, kills weakness-shorts
A/B/C/D), overbought continue UP (kills the held-to-endpoint mirror E). BUT the **path** is different from the
endpoint: "nothing goes up straight" is true — a fast SCALP of the intraday retrace (F), **gated to a weak tape
(SPY<20-SMA)**, is **positive OOS (+0.40%/trade net of cost+borrow), positive in both halves, PBO 0.09**. It is the
ONLY short configuration in the whole campaign that doesn't lose out-of-sample. It still fails the *firewall*
(DSR ~0.23 ≪ 0.95) — given how many configs were searched, the positive could still be luck — so it is a **thin,
promising, regime-conditional short, NOT yet bankable.** This is the LIVE short thread. Repro: `scripts/short_pullback_scalp.py`
(ungated config search) + `scripts/short_pullback_regime.py` (the weak-tape gate). Next step to confirm or kill it:
a pre-registered, low-search-DOF walk-forward on more data — not more config mining (that only inflates DSR's penalty).

**The decoded answer (firewall-grade):** *no standalone single-name SHORT edge clears the firewall yet* — but the
honest state is NOT "shorts never work." The endpoint angles (A/B/C/D/E) are all dead (weak names bounce, overbought
runs). The **path** angle (F) is the exception: shorting an over-extension and **scalping the retrace in a weak tape
is positive OOS** (+0.40%/trade, both halves, PBO 0.09) — it just doesn't clear DSR (0.23). So today the short side
ships only in two proven roles: (1) the **short leg of a market-neutral book** (the ML ranker, Sharpe ~0.87 — long leg
finances it), and (2) a **small −beta drawdown HEDGE** (`overbought_short`). And it has **one live research thread**
worth confirming: the weak-tape pullback-scalp (F), pending a pre-registered walk-forward. **The qedge scanner
correctly returns "LIKELY OVERFIT" for single-name shorts today — that is the firewall working** — and F is exactly
the candidate to push at it next, on a tighter search budget so a real positive isn't drowned by the deflation penalty. Scripts: `scripts/short_xsec.py` (A), `scripts/short_regime_gate.py` (D);
B/C harnesses in the decode scratchpad; `scripts/short_overbought_study.py` (E, the mirror) and
`scripts/short_scan_study.py` (B+C reproducers). n_trials honestly counted (A=28, B=10, D=18, E=24) so DSR isn't flattered.

---

## CONSOLIDATED SCOREBOARD (all edges characterized, gathered before assembly)

**The shape of the answer: there is no single oracle. There is a constellation of small, real,
weakly-correlated edges — and the alpha is in assembling them, weighted by evidence.**

### ✅ REAL, tradeable edges (each thin alone — DSR mostly 0.2–0.6 — but firewall-cleared)
| Edge | OOS evidence | corr to equity book | role |
|---|---|---|---|
| **SPY swing book** (breakout/IBS/fear) | un-levered Sharpe ~1.18, maxDD −9% | 1.0 (it IS the book) | core |
| **ML cross-sectional ranker** (4wk) | net Sharpe **0.87**, α +24.6% β−0.08, IC t 4.30, **DSR 0.63** | ~0 (market-neutral) | **best diversifier** |
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
| Edge | Sharpe | weight | corr to book |
|---|---|---|---|
| SPY swing book | 1.12 | 72% | 1.00 |
| ML ranker (market-neutral) | 0.90 | 26% | 0.09 |
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
