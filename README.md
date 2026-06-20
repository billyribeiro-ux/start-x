# Start-X — Self-Learning Swing-Trading Research Engine

Reconstructs **why** a stock spiked or sold off over any date range (causal attribution /
event study on FMP data), then learns which catalysts reliably precede big moves.

> **Guiding principle:** success = *out-of-sample, cost-adjusted robustness*, never in-sample
> win rate. A scan that looks ~100% "perfect" on history is treated as an **overfitting red
> flag**, not a goal. We trust only edges that survive data the model never saw.

## Status
- **Phase 0** — data spine: rate-limited FMP client + Parquet cache ✅
- **Phase 1** — the engine: event detection → event study (CAR) → causal attribution →
  Streamlit *Move Explorer* + first self-learning (catalyst-reliability) ✅
- Phases 2–7 (features, labels, ML, purged-CV validation, backtest, forward test) — next.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env          # then put your FMP key in .env (gitignored)
```
If your FMP key was ever exposed, **regenerate it** and only ever store it in `.env`.

## Use
```bash
# Backfill prices + catalysts for the seed universe into the local cache
python scripts/backfill.py

# Launch the Move Explorer dashboard
streamlit run src/startx/dashboard/app.py
```
Pick a symbol + date range → see every statistically significant move, the attributed
cause(s) with confidence, the event-study average move profile, and the learned
catalyst-reliability table.

## Seed universe
`SPY, SPX (^GSPC), QQQ, IWM` (macro-driven) and `TSLA, NVDA, AAPL` (company-catalyst-driven).
Horizons: short-term 1–10 trading days and long-term weeks–months. Edit `config/universe.yaml`
to expand.

## Layout
`src/startx/fmp` client/endpoints · `…/data` cache, prices, universe · `…/events` detect,
study, attribute, engine, learn · `…/dashboard` Streamlit. Analytics are pure-Python services
returning DataFrames/dicts so a FastAPI + Svelte5 frontend can consume them later unchanged.
