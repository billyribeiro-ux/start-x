# Findings — Swing-Trading Research (SPY/SPX)

What we learned, hard-won and evidence-backed. Read this before re-deriving anything.

## The big result
- **`prob_up` (the ML directional model) is a coin flip.** OOS AUC ~0.50 on SPY. It looked good
  in-sample (2023-now) and **collapsed to breakeven out-of-sample (2018-2022, net ~$82/$10k).**
  Predicting next-move direction on the S&P from this data is not a real edge.
- **A rule-based mean-reversion entry beats it AND survives out-of-sample.** Buy RSI(2)<15 dips
  while above the 200-day (Connors-style), exit at a horizon-matched ~1.5 ATR target/stop:
  - OOS 2018-2022: 340 trades, **57% launch, 59% win, +$9,379/$10k** — *same* expectancy as
    in-sample (2023-now: 60%/61%). Consistency across regimes = a real edge.
  - Jan 2025 -> Jun 2026 (SPY+SPX): 49 trades, **63% win, +17.5%, PF 1.53, maxDD -11.7%.**
- Code: `src/startx/strategy/mean_reversion.py`.

## Exit / stop lessons (all tested, several counter-intuitive)
- **The fixed 1.5σ triple-barrier target capped winners — we captured only ~39% of the move.**
  After a target exit, price ran another ~+2% 75% of the time.
- **Let winners run (chandelier/exhaustion) tripled the in-sample $... but FAILED out-of-sample.**
  It was tuned to the 2023-24 bull; it lost on 2018-2022. Regime-dependent, not a universal edge.
- **3 ATR was the wrong target for a 1-10 day swing.** SPY's typical 10-day move is ~1.4 ATR, so a
  3 ATR target asks for a 2-3 week move — that's why ~78% of trades "died". **Size targets to the
  expected move (~1-1.5 ATR for a short swing), not an arbitrary number.**
- **Tightening to "protect" cuts the runners and LOSES money:** +1 ATR lock and breakeven stops
  both cut net P&L (you tap out runners that pull back before they run). Correct relabel: a +1 ATR
  exit is a small WIN and a breakeven is a SCRATCH (not a loss) — but the $ total still drops.
- **A looser stop (2.5-3σ) helps the recent bull window but still loses OOS** — it's a *bet that
  the uptrend continues*, not a free lunch.

## VIX as a signal (the long debate — settled by the tape)
- **VIX "neutral" price ≈ 18.** Three independent methods converge: 16y mean = 18.4; the
  mean-reversion attractor (zero-drift level from `dVIX = 0.695 - 0.0378*VIX`) = **18.4**; it's the
  center of the distribution (median 16.8). Below ~18 VIX drifts up, above ~18 it mean-reverts down
  (half-life ~18 trading days).
- **High VIX is NOT a short on the index — it's a bounce.** 2010-2026: shorting SPX at VIX>=30
  loses −0.95%/trade (35% win); VIX>=40 loses −2.13% (17% win). Forward S&P *rises* with VIX at
  every horizon (VIX 30-40 -> +1.30% fwd 10d). The −0.75 VIX/SPX relationship is **same-day
  (coincident), not predictive** — VIX is high *because* the drop already happened.
- **Capitulation (persistent vol breakout) = a LONG.** ≥3 consecutive closes above VIX's 2.5σ
  Bollinger band while VIX>neutral -> SPX +0.65-0.89% fwd 5d, ~68-70% up. Shorting it wins ~30%.
  The market **gaps up ~+0.28% the next day (10x the +0.03% baseline, 63% up)** — so enter at the
  signal close to own the gap. Module: `src/startx/strategy/vix_capitulation.py` (long index,
  chandelier 3 ATR). Full 2010-now: 19 trades, PF 2.26, +$124/sh; **survives OOS (2010-2022 PF
  1.22)** where the dip-buy chandelier did not — but it's rare (~1.2/yr) and small-sample, and it
  fails buying into the *first leg of a bear* (Jan 2022, −$19.75). Needs a trend/regime guard.

## Loser anatomy (the "died at stop" trades)
- ~15% straight losers (never green), ~45% worked but <1 ATR (chop/fakeouts), ~38% hit >=1 ATR
  then round-tripped. Only the last group is salvageable by stop management, and doing so costs
  more in cut winners than it saves.
- **Straight shooters (clean 3 ATR launchers) announce themselves: Day-1 close ~+0.8 ATR green,
  entered on a volume surge into weakness (capitulation).** Grinders close Day-1 red/flat. A
  *post-entry* Day-1 confirmation is the supportable filter (the entry bar itself can't predict it).

## Self-learning meta-filter on the IBS dip-buy (the loss-autopsy doing its job)
Ran the project's own `autopsy_trades → win_loss_signature → walk_forward_metalabel` on the IBS
dip-buy (885 trades pooled SPY/SPX/QQQ for training; SPY 2019-26 evaluated). The machinery **learned
to skip losers** instead of us hand-coding a rule.
- **Leak caught in our OWN module:** `forensic_feature_columns` was a deny-list and let `ret`/
  `exit_price`/price levels through → OOS AUC 1.0, fake "100% win". Fixed to an allow-list (forensic
  families + candle one-hots + registered pre-entry signals); regression test added. THE project
  principle catching itself.
- **What it learned:** winners dip on a wider-range / ATR-expanding bar that gaps slightly UP (a
  flush already snapping back); losers are quiet bleeds-lower (negative gap, gap-down marubozu).
  Every single forensic is weak alone (AUC 0.46-0.57) — the edge is only in the combination.
- **OOS lift (SPY TEST 2023-26, threshold P(win)≥0.50 pre-registered):** base IBS 69 trades / 65%
  win / PF 1.93 / drift-adj alpha **−0.055%** → filtered 37 / **84%** / PF **5.12** / maxDD −1.8% /
  drift-adj alpha **+0.081%**. Flips alpha negative→positive; skips the 2026-03 knife-catch cluster
  (all three < 0.50). Honest caveats: small n (37 test / 7 in 2025+), edge is model-dependent
  (GradientBoosting barely filters) and threshold-sensitive — loss-avoidance, not a money-printer.

## Relearned signal set — 2019→2026 (TRAIN 2019-22 / TEST 2023-26, drift- & risk-adjusted)
Four parallel agents re-derived signals from scratch on the locked window. The honest metric is
**alpha = per-trade expectancy − same-holding-period SPY drift** (a part-time long silently collects
the bull; subtract it). On *total return* nothing beats buy-and-hold (you're <100% invested in a
+96% bull); the edges below are **risk-adjusted** — ~¼ the drawdown, higher ret/DD.

**SURVIVORS (real OOS edge):**
- **20-day-high breakout > SMA200, chandelier exit** (momentum) — best risk-adjusted: TEST PF 3.64,
  maxDD −4.4%, ret/DD 11.8 vs 5.1 B&H. Generalizes untuned to QQQ/SPX; **fails on IWM**. Module:
  `strategy/momentum_breakout.py` (TEST 2023-26: 20 trades, 55% win, PF 3.78, +50%, −4.8% DD).
- **VRP-high** (VIX − 20d realized vol in top 5% of trailing yr), chandelier (volatility) — TEST
  18 trades / 61% / PF 5.36, decent n, consistent. Best-sampled vol entry.
- **IBS < 0.1 in an uptrend**, fixed 1.5 ATR (mean-reversion) — only oversold signal with positive
  drift-adjusted alpha across SPY/SPX/QQQ/IWM (+0.05 to +0.73%), 62–69% win. Most trades of any.
- **VIX-capitulation long** (≥3 closes > 2.5σ band, VIX>neutral), chandelier — holds OOS (TEST
  ~80%/PF 15) but **tiny n (4–5)**: high-conviction, low-frequency.
- **Risk filters:** credit-on (HYG/LQD>SMA50) and gold-calm (GLD/SPY<SMA20) roughly **halve
  drawdown**; on a dip-buy, credit-on lifts win 70%→77%, PF 1.79→3.03. *Caveat: intermarket cache
  only starts 2022-06 → single-window test, soft.*

**ADAPTIVE VIX NEUTRAL = zero-drift attractor on trailing 252d** (`adaptive_neutral`, default).
Static 18 was a trap (leaned on one lucky 2021 trade); p70-252 is the simpler near-equal fallback.

**DEAD / discarded OOS:** RSI-2 (raw-positive but **drift-adjusted alpha −0.22%** — was just riding
drift), pre-FOMC drift (didn't persist), Santa/sell-in-May (B&H proxies), day-of-week, rates
direction, dollar, breadth%, growth-vs-small-cap (all TRAIN→TEST collapses / 2022 artifacts).

## Market internals (breadth derived from the 503 S&P constituents)
No vendor breadth feed on FMP, so internals are derived from the constituent panel (% above 50/200d,
A/D, up/down volume, new highs−lows, McClellan), point-in-time. Two honest results:
- **Breadth *confirmation* for the FEAR sleeve does NOT generalize.** The 2026-03 "wait for the
  washout climax" story was a cherry-pick (n=13 fear trades; breadth barely separates winners from
  losers, and the gate slightly *hurt* the book OOS). Not shipped.
- **Breadth *guard* on the IBS dip sleeve IS real** — and it's a risk filter, not alpha. IBS losers
  buy oversold dips into a **broad breakdown** (up-volume ≤20% / ≥80% down-volume, A/D collapsing);
  up/down volume separated IBS win/loss at **AUC 0.78**. Standing aside on those days: win 58→61%,
  **maxDD −19.4%→−15.5%** (FULL 2019-26), vetoes the 2026-03-18 falling knife; small raw-return
  give-up on the calm 2023-26 window (its value shows up in stress). Module:
  `strategy/market_internals.py` (`not_breaking_down`). Internals also feed the human thesis layer.

## Cross-desk stress test — Quant / ML / Mathematician / Market-Maker (2019-26)
Four specialist agents independently stress-tested the book. They converge hard:
- **The system's value is RISK-ADJUSTED, not alpha over the index.** Best portfolio (concurrent,
  vol-targeted equal-risk 3%/stop + gold-calm overlay + −10% drawdown circuit-breaker): Sharpe
  **1.43 / Calmar 1.05 / maxDD −16%** vs SPY B&H 0.85 / 0.46 / −34%, and strong in BOTH halves. But
  raw CAGR ~ties B&H. The real edge is **diversification** (sleeve corr 0.09–0.52); book DSR 0.995.
- **Only `momentum_breakout` has real, durable, cost-robust drift-adjusted alpha** (+0.17%,
  P(α≤0)=0.028; fill-insensitive; durable premium = trend-chasers + disposition effect). Trust most
  — but underpowered solo (DSR 0.46), so it lives inside the book.
- **IBS = real, robust *pattern* but ~0 alpha** (exposure timing; dies in <1bp slippage / next-open).
- **VIX-capitulation = likely NOISE / beta not alpha** (DSR 0.10; drift-alpha −1.22%, P(α≤0)=0.985;
  profit lives in 5 lucky trades; the "PF 15 / 80% win" was 5 observations). **Demoted to unproven.**
- **The earlier IBS meta-filter (84% / PF 5.12) is FRAGILE** — purged-CV AUC <0.5, PBO 0.83, inverted
  OOS calibration. Its only honest effect is drawdown-smoothing via pooling, NOT alpha. ML take/skip
  gates and learned/exhaustion exits do NOT beat the rules (they cut the home-runners).
- **Chandelier 3.0 ATR exit is mathematically near-optimal & robust** (EV-optimal given median MFE
  ≈4–5 ATR; "wider is better" is in-sample drift, not significant). **IBS should move off its fixed
  1.5 ATR target to the chandelier** (leaves ~0.5–0.9 ATR/trade on the table now).
- **VIX term structure does NOT rescue the short thesis** (backwardation not robustly predictive; no
  tradeable caution) — fear extremes are bounces (4th confirmation). **NEW gauge: VVIX (vol-of-vol)
  ≥ p90** — fwd-5d +1.3%/+1.1% TRAIN/TEST, 76% up both, partly orthogonal to spot VIX, better sample
  (n=37 vs capitulation's 9) → a better *trigger* for the fear sleeve than spot VIX.
- **Discarded OOS:** credit-on overlay, regime-tilt, vol-target, learned/exhaustion exits, ML
  take/skip. Options/skew data unavailable on this FMP plan.

## Standing lessons (the ones that change how we trade)
- **Coincident ≠ predictive.** VIX/SPX is −0.75 *same-day* — a mirror, not a forecast. The edge
  only lives in relationships that survive the shift from same-bar to next-bar. Most "obvious"
  market reads are coincident and untradeable.
- **The tradeable edge is usually the *opposite* of the naive read.** Fear at an extreme isn't a
  sell — it's the tell that the sellers are done. High VIX = bounce, not short.
- **"High/low" is meaningless without an *adaptive* baseline.** A static full-sample VIX neutral
  (18) is wrong: the baseline drifts hard by regime (median VIX 10.8 in 2017, 14.6 in 2024, 18.3 in
  2026). Use a trailing/rolling neutral, not a frozen constant. The same goes for any threshold.
- **The exit carries the edge as much as the entry.** The capitulation signal only profits with a
  chandelier; fixed targets lose OOS because the P&L lives in a few monsters. Don't cut runners.
- **Re-learn signals when the window changes.** A shorter/newer period is a *different* regime — do
  not carry an old edge (e.g. RSI-2) forward as gospel; re-discover and re-rank on the new data.

## Method guardrails (do not "fix" away)
- Success = **out-of-sample** robustness, not win rate or in-sample $. Every flashy in-sample
  number this project produced was a mirage until tested on a held-out period.
- Penalize the deflated Sharpe / be honest about the number of configs tried (multiple testing).
- Beating buy-and-hold on the S&P is genuinely hard; "makes money" != "beats the index".

## Open / next
- Layer the validated mean-reversion base with the Day-1 confirmation + volume capitulation filter
  (validate each OOS).
- Test complementary strategies (momentum/breakout) with the same rigor.
- Expand beyond SPY/SPX to the catalyst-driven movers, where edges are richer.

## Execution & leverage drill (3 parallel agents, 2019-26)
- **The book survives realistic (next-open) execution at the portfolio level** (Sharpe/PF/maxDD
  ~unchanged, OOS too) — it is **trend-capture, not entry precision**. But every sleeve buys a
  +3-8 bp overnight gap-up, so on a per-trade drift-adjusted basis the entry signals have NO
  standalone edge off the close print (IBS −0.16%, t=−2.01; breakout mildly negative). Do not try to
  scale entry frequency or shorten holds.
- **Un-levered, the book makes ~half SPY's CAGR with a quarter of the drawdown** (1.0×: 8.8% CAGR /
  −9.1% maxDD / Sharpe 1.18 / Calmar 0.96 vs SPY 15.8% / −34.1% / 0.85 / 0.46). The edge is **risk
  reduction, not raw return**; SPY's nominal lead is pure bull beta the book declines to take.
  Leverage is Sharpe-neutral and just amplifies — **1.5× is the Calmar sweet spot, 3× unjustified.**
- **R-multiple:** expectancy +1.1 to +1.3 R/trade (lose ~1R, win ~3.4R, ~50% win).
