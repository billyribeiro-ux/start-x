# TRIALS LEDGER — every configuration tested, with N fed to the deflation bar

Per the Operating Standard (v2): breadth is only safe if every trial is counted. `N` is the running
count of distinct configurations searched; it feeds the Deflated Sharpe / PBO bar so the threshold to
declare an edge rises automatically as we explore. **A Sharpe with no N attached is not a result.**
Real-data window is **2018-01-01 → 2026-06-26** (FMP/cache coverage is ~96-100% here; pre-2016 is
survivorship-contaminated and its numbers are void — see the membership coverage report). Newest first.

Hypothesis gate (must be answerable before a method is tried): (1) economic rationale, (2) who is on
the other side and why they keep losing, (3) the pre-stated kill condition.

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
Next (P5): gate the signal on regime (crisis/risk_off) and re-validate OOS as a NEW trial (adds to N), +
SHAP/permutation feature promotion + drift monitor. Scripts: scripts/swing_validate.py.

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
Short campaign ≈ 336 · ranker A/B = 4 · misc baselines ≈ 10 → **N ≈ 350** and counting. With N this large,
the deflation bar is high by design: only edges with a strong prior + clean OOS + low search cost clear it.
Validated production systems (the three rule-based books, the better long model) were each established under
their own pre-registered, low-N protocols — see CHANGELOG/STRATEGY/FINDINGS — and are NOT diluted by this
exploratory N. New exploratory trials add to N here; pre-registered confirmations are logged separately.
