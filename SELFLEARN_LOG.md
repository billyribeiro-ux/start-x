
## Iteration 1 — 2026-06-28
- taken 366; overall loss rate 52%.
- WHY: worst trigger `pullback` (57% loss); top loss-separating feature `rsi2`.
- tested `feature_tail:rsi2:<` -> rejected (no firewall clear; streak 1).
- ML loss-veto (walk-forward, veto_q=0.85, same scored subset): exp +1.21% -> +1.04%, DSR 0.85 -> 0.61 (vetoed cohort exp +2.22% — the 'avoid' trades were PROFITABLE).
- standing avoid set: 0 rule(s); holdout-equivalent exp +0.95% -> +0.95%.

## Iteration 2 — 2026-06-28
- taken 366; overall loss rate 52%.
- WHY: worst trigger `pullback` (57% loss); top loss-separating feature `rsi2`.
- tested `trigger:pullback:==` -> rejected (no firewall clear; streak 2).
- ML loss-veto (walk-forward, veto_q=0.85, same scored subset): exp +1.21% -> +1.04%, DSR 0.85 -> 0.60 (vetoed cohort exp +2.22% — the 'avoid' trades were PROFITABLE).
- standing avoid set: 0 rule(s); holdout-equivalent exp +0.95% -> +0.95%.

## Iteration 3 — 2026-06-28
- taken 366; overall loss rate 52%.
- WHY: worst trigger `pullback` (57% loss); top loss-separating feature `rsi2`.
- tested `feature_tail:dist_sma200:>` -> rejected (no firewall clear; streak 3).
- ML loss-veto (walk-forward, veto_q=0.85, same scored subset): exp +1.21% -> +1.04%, DSR 0.85 -> 0.60 (vetoed cohort exp +2.22% — the 'avoid' trades were PROFITABLE).
- standing avoid set: 0 rule(s); holdout-equivalent exp +0.95% -> +0.95%.

## Iteration 4 — 2026-06-28
- taken 366; overall loss rate 52%.
- WHY: worst trigger `pullback` (57% loss); top loss-separating feature `rsi2`.
- candidates exhausted; avoid set converged.
- ML loss-veto (walk-forward, veto_q=0.85, same scored subset): exp +1.21% -> +1.04%, DSR 0.85 -> 0.60 (vetoed cohort exp +2.22% — the 'avoid' trades were PROFITABLE).
- standing avoid set: 0 rule(s); holdout-equivalent exp +0.95% -> +0.95%.
