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

## Loser anatomy (the "died at stop" trades)
- ~15% straight losers (never green), ~45% worked but <1 ATR (chop/fakeouts), ~38% hit >=1 ATR
  then round-tripped. Only the last group is salvageable by stop management, and doing so costs
  more in cut winners than it saves.
- **Straight shooters (clean 3 ATR launchers) announce themselves: Day-1 close ~+0.8 ATR green,
  entered on a volume surge into weakness (capitulation).** Grinders close Day-1 red/flat. A
  *post-entry* Day-1 confirmation is the supportable filter (the entry bar itself can't predict it).

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
