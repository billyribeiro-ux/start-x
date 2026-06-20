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
