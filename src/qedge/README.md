# qedge — a leakage-hardened quant-research scanner

`qedge` is a self-contained edge-discovery scanner that implements the **Quant
Research Operating Contract** (López de Prado / Ernie Chan standard). Its governing
philosophy is adversarial: **every strong backtest is guilty of leakage until
proven innocent.** It lives inside the `start-x` repo as a top-level package under
`src/qedge/`, reuses the proven `startx` methodology where that already meets the
contract, and implements the rest fresh.

It runs the whole stack — point-in-time data → point-in-time features → regime →
triple-barrier labels → purged CPCV → cross-family ensemble → execution costs →
the Deflated-Sharpe / PBO survival gate → capacity — and returns a single honest
verdict per symbol: **ROBUST** or **LIKELY OVERFIT**.

---

## Table of contents
1. [Quickstart](#quickstart)
2. [Choosing the analysis period](#choosing-the-analysis-period)
3. [Install & setup](#install--setup)
4. [What it does (methodology)](#what-it-does-methodology)
5. [Architecture & package layout](#architecture--package-layout)
6. [Features](#features)
7. [Data sources & honest limitations](#data-sources--honest-limitations)
8. [Output: EdgeResult, scorecard, run records](#output-edgeresult-scorecard-run-records)
9. [Configuration reference](#configuration-reference)
10. [Testing & verification](#testing--verification)
11. [Extending to intraday / other bar intervals](#extending-to-intraday--other-bar-intervals)
12. [Known findings & caveats](#known-findings--caveats)

---

## Quickstart

```bash
# From the repo root. Default window is 2018-01-01 -> 2026-06-18.

# Offline, deterministic (synthetic data — no API key, byte-reproducible):
python -m qedge.scanner.cli --universe SPY,QQQ,IWM --horizon short

# Live, against real FMP data (reads FMP_API_KEY from the gitignored .env):
python -m qedge.scanner.cli --universe SPY,QQQ,IWM --horizon short --live
```

Example live scorecard (real SPY/QQQ/IWM, 2018-01-01 → 2026-06-18):

```
| Symbol | Horizon | Events | OOS Sharpe | Deflated Sharpe | PBO   | Verdict               | Break-even AUM | Top-3 Features                | Unpopulated Dims        |
| SPY    | short   | 1928   | 0.012      | 0.700           | 0.457 | [FAIL] LIKELY OVERFIT | -              | volvol_21_63, overnight_gap…  | options_nbbo, internals |
| QQQ    | short   | 1928   | 0.022      | 0.667           | 0.229 | [FAIL] LIKELY OVERFIT | -              | fracdiff_logclose, volvol…    | options_nbbo, internals |
| IWM    | short   | 1928   | 0.008      | 0.321           | 0.457 | [FAIL] LIKELY OVERFIT | -              | semidev_63d, fracdiff_logclose| options_nbbo, internals |
```

A `-` in **Break-even AUM** means the edge did not clear the gate, so its capacity
ceiling is reported as N/A (capacity is only meaningful for a shippable edge).

---

## Choosing the analysis period

The scan window is a first-class, caller-selectable parameter. **Default:
`2018-01-01` → `2026-06-18`** (inclusive bounds).

**CLI:**
```bash
python -m qedge.scanner.cli --universe SPY,QQQ,IWM \
    --start 2020-03-01 --end 2022-12-31 --live
```
`--start` / `--end` take `YYYY-MM-DD` and are validated (a malformed date exits
with a usage error). When omitted they fall back to `cfg.data.analysis_start` /
`cfg.data.analysis_end`.

**Programmatic:**
```python
from qedge.scanner.pipeline import run_scan, scan_symbol
from qedge.data.fmp_adapter import FMPPriceFeed

# whole universe, custom window
results = run_scan(["SPY", "QQQ", "IWM"], feed=FMPPriceFeed(),
                   horizon="short", start="2018-01-01", end="2026-06-18")

# one symbol
edge = scan_symbol("SPY", feed=FMPPriceFeed(), horizon="short",
                   start="2018-01-01", end="2026-06-18")
print(edge.verdict, edge.deflated_sharpe, edge.pbo)
```

**Change the default** in `src/qedge/config.py`:
```python
class DataConfig(BaseModel):
    analysis_start: str = "2018-01-01"
    analysis_end:   str = "2026-06-18"
```
or override per-run with the env var prefix, e.g. `QEDGE_DATA__ANALYSIS_START=2015-01-01`.

> **How the window is enforced (and a gotcha that was fixed).** The upper bound is
> a point-in-time guarantee (a daily bar is public at its close, so only rows with
> `date <= end` are returned). The lower bound is enforced **on the adapter output**,
> not just forwarded to the fetch — because the reused `startx` price cache keys by
> symbol only (no date range), a frame cached under an earlier request would
> otherwise be served wholesale. The default price source also **refreshes from FMP
> when the cache does not reach back to the requested `start`**. Verified: SPY/QQQ/IWM
> all fetch exactly `[start, end]` (2,127 bars over 2018-01-02 → 2026-06-18).
>
> Note: the longest feature warm-up is ~126 bars, so the first ~6 months after
> `start` produce NaN/dropped rows; effective analysis begins after warm-up.

---

## Install & setup

- **Python 3.11.**
- **Dependencies** are exactly pinned in `requirements-qedge.txt`:
  ```bash
  python -m pip install -r requirements-qedge.txt
  ```
  Core (numpy 1.26.4, pandas 2.2.2, scipy 1.13.1, statsmodels 0.14.2, pydantic) plus
  the modeling stack (scikit-learn 1.5.2, lightgbm 4.5.0, shap 0.46.0, hmmlearn 0.3.2,
  river 0.21.2, ruptures 1.1.9) and tooling (pytest, mypy, ruff). Also available as
  the `qedge` optional-dependency group in `pyproject.toml`.
- **FMP key** (for `--live` only) goes in the gitignored `.env` at the repo root:
  ```
  FMP_API_KEY=your_key_here
  ```
  It is read via the reused `startx` settings layer and is never committed.

---

## What it does (methodology)

Every stage obeys the contract:

- **Point-in-time correctness** — each feed takes an explicit `asof` and returns only
  information public at or before it; every feature declares an
  `InformationBoundary`. The no-lookahead guarantee is enforced by a
  **registry-wide canary test** (feed poisoned future rows; assert the value at `t`
  is byte-identical).
- **Triple-barrier labeling** (López de Prado) — `short` (10-day vertical, 1.5×vol
  barriers) and `long` (63-day, 2.5×vol) presets; emits `t1` for purging. *(Reused
  from `startx.labeling`.)*
- **Meta-labeling** — a walk-forward P(win) gate with a strict `t1 <= entry` firewall.
- **Uniqueness sample weights** — overlapping labels are down-weighted by concurrency.
- **Fractional differentiation** — fixed-width FFD (`features/fracdiff.py`) plus an
  ADF min-`d` search; the FFD fixed backward window is the PIT guarantee. Exposed as
  the `fracdiff_logclose` feature.
- **Validation (non-negotiable):** Combinatorial Purged CV with embargo (no random
  k-fold), **Deflated Sharpe Ratio** and **Probability of Backtest Overfitting**
  reported on every scan, deflated by the **multiple-testing trial count** (a shared
  `TrialLedger` counts every symbol/horizon evaluated). **Survival gate:
  `passed = DSR >= 0.95 AND PBO <= 0.5`.**
- **Feature importance** — clustered **MDA** and **clustered-SHAP** on **out-of-sample**
  folds only; in-sample MDI is banned; correlated features are clustered first.
- **Regime detection (first-class)** — a Gaussian **HMM** whose state is assigned by
  **forward-filtering** (`predict_pit`), never Viterbi/forward-backward smoothing
  (which would leak the future); plus change-point (`ruptures`) and vol/correlation
  clustering. Inputs are standardised with fit-time stats for clean convergence.
- **Online learning & drift** — river incremental learners with explicit **ADWIN** and
  **Page-Hinkley** concept-drift detectors.
- **Ensemble** — soft-voting across diverse families (RandomForest + LightGBM +
  calibrated logistic); diversity over single-model tuning. Calibration degrades
  gracefully (adapts folds to the minority class) on thin event sets.
- **Execution realism (before any Sharpe)** — commission/slippage, partial fills
  (capped at a share of bar volume, remainder carried), and an explicit options
  bid-ask model with NBBO-aligned fills.
- **Capacity** — square-root market-impact decay → the **break-even AUM** at which the
  edge dies.

---

## Architecture & package layout

```
src/qedge/
  config.py            # pydantic-settings: every tunable (no magic numbers)
  repro.py             # deterministic seeds, sha256 data-snapshot hashing, run records
  data/
    protocols.py       # PIT Protocols: PriceFeed, FundamentalsFeed, OptionsNBBOFeed,
                        #   InternalsFeed, NewsFeed (+ FeedNotAvailable)
    boundary.py        # InformationBoundary — each feature's exact info boundary
    fmp_adapter.py     # FMP-backed feeds (injectable source; PIT-clipped; cache-refresh)
    stub_adapters.py   # UnavailableOptionsNBBOFeed / UnavailableInternalsFeed (raise)
    synthetic.py       # deterministic SyntheticMarket implementing all 5 Protocols
    universe.py        # universe resolution + survivorship-aware membership hook
  labeling/__init__.py # wrappers over startx triple_barrier / weights / metalabel
  features/
    base.py            # FeatureSpec + FeatureRegistry (@feature) + compute_features
    technical.py       # the 16 registered trailing features
    fracdiff.py        # fractional differentiation + ADF min-d
    cluster.py         # correlation-distance feature clustering (pre-importance)
  validation/__init__.py  # make_cpcv, survival_gate (DSR+PBO), TrialLedger
  modeling/
    importance.py      # OOS MDA + clustered-SHAP
    regime.py          # HMM (forward-filtered) + change-point + vol/corr clustering
    online.py          # river online learner + ADWIN/Page-Hinkley drift
    ensemble.py        # cross-family soft-voting + startx return-stream combiner
  execution/
    costs.py           # costs / slippage / partial fills / latency
    options.py         # options bid-ask + NBBO-aligned fills
    capacity.py        # market-impact decay -> break-even AUM
  scanner/
    pipeline.py        # the adversarial harness: ScanConfig, scan_symbol, run_scan, EdgeResult
    report.py          # scorecard_table, results_to_frame, format_edge
    cli.py             # python -m qedge.scanner.cli
tests/qedge/           # mirrors the package; 231 passed / 2 skipped (live FMP)
```

Reused `startx` modules (imported, not duplicated): `labeling.triple_barrier`,
`labeling.weights`, `learning.metalabel`, `validation.{cpcv,purged_cv,metrics,walkforward}`,
`models.selection`, `data.pit`, `portfolio.ensemble`, `portfolio.execution`,
`fmp.client`/`endpoints`, `data.prices`/`cache`.

---

## Features

16 pure, trailing features are registered at import in
`features/technical.py`; the registry-wide canary proves each is no-lookahead:

| Feature | Meaning | Warm-up (bars) |
|---|---|---|
| `ret_1d`, `ret_5d`, `ret_21d` | trailing simple returns | 1 / 5 / 21 |
| `rvol_21d`, `rvol_63d` | annualised realized vol | 22 / 64 |
| `mom_63d`, `mom_126d` | momentum (quarter / ~6-month) | 63 / 126 |
| `rsi_14` | Wilder RSI | 15 |
| `ibs` | internal bar strength (close-low)/(high-low) | 0 |
| `overnight_gap` | open / prior close − 1 | 1 |
| `dist_sma_100` | close / SMA(100) − 1 (trend stretch) | 100 |
| `volvol_21_63` | vol-of-vol | 84 |
| `semidev_63d` | downside semi-deviation | 64 |
| `atr_pct_14` | ATR(14) / close | 15 |
| `skew_63d` | trailing return skewness | 64 |
| `fracdiff_logclose` | fixed-width FFD of log close (stationary w/ memory) | ~window of FFD weights |

Add a feature by writing a pure `df -> Series` function and registering a
`FeatureSpec(name, fn, boundary)` — the canary test then covers it automatically.

---

## Data sources & honest limitations

**FMP provides equity OHLCV + fundamentals/earnings + news.** It does **not** provide
OPRA options flow, real-time TICK/TRIN/ADD/VOLD internals, or tick/order-flow.

The contract still models those dimensions, honestly:
- `OptionsNBBOFeed` and `InternalsFeed` are real typed Protocols. The FMP path uses
  `UnavailableOptionsNBBOFeed` / `UnavailableInternalsFeed`, which **raise
  `FeedNotAvailable`** — never fabricated values.
- Every scorecard lists these as **unpopulated dimensions**, so a reader never
  mistakes a partially-probed scan for a fully-exercised one.
- **Survivorship:** the seed universe is index ETFs (which persist), so survivorship
  bias is minimal. `universe.py` carries a `members(asof)` hook so a future
  delisted-aware single-name universe drops in.

For fully-deterministic, offline runs (tests, demos, reproducibility), the
`SyntheticMarket` implements **all five** feeds from a seeded generator.

---

## Output: EdgeResult, scorecard, run records

`scan_symbol` returns a frozen, serialisable `EdgeResult`:

| Field | Meaning |
|---|---|
| `symbol`, `horizon` | what was scanned |
| `n_events` | clean labelled events tested |
| `n_trials` | running multiple-testing count (deflates the DSR) |
| `oos_sharpe` | out-of-sample Sharpe of the take/skip stream |
| `deflated_sharpe`, `pbo` | the two survival metrics |
| `passed`, `verdict` | `True`/`ROBUST` vs `False`/`LIKELY OVERFIT` |
| `break_even_aum_usd` | capacity ceiling (NaN/unbounded handled by the report) |
| `top_features` | OOS clustered-MDA ranking |
| `unpopulated_dimensions` | contract dimensions with no feed |
| `feature_snapshot_hash`, `run_id` | reproducibility provenance |

The CLI prints `scorecard_table(...)` and writes one **run-record JSON** per symbol
(`repro.write_run_record`) under `--out` (default `cfg.scanner.output_dir`),
stamping the config, data hashes, metrics, and the chosen analysis window — a
reproducible audit trail (same inputs → same `run_id`).

---

## Configuration reference

All knobs live in `src/qedge/config.py` (env override prefix `QEDGE_`, nested with
`__`, e.g. `QEDGE_VALIDATION__DSR_MIN=0.9`). Highlights:

- `data.analysis_start` / `analysis_end` — default scan window (`2018-01-01` / `2026-06-18`).
- `data.universe` — `("SPY", "QQQ", "IWM")`.
- `labeling.*` — short/long triple-barrier presets (10-day / 63-day).
- `validation.cpcv_n_groups=6`, `cpcv_n_test_groups=2`, `embargo_pct=0.01`,
  `dsr_min=0.95`, `pbo_max=0.5`.
- `features.fracdiff_*` — FFD `d` grid, ADF p-value, the `fracdiff_feature_d` (0.5).
- `regime.hmm_n_states=3`, `hmm_n_iter=200`, change-point penalty.
- `online.adwin_delta`, `page_hinkley_*`.
- `execution.*` — commission/slippage bps, partial-fill participation,
  `intraday_latency_ms`, `options_half_spread_bps`.
- `capacity.*` — ADV-participation grid, impact coefficient, AUM grid.
- `scanner.seed=7`, `output_dir`.

---

## Testing & verification

```bash
# Full suite — run SINGLE-THREADED (the pipeline tests are heavy; concurrent
# heavy runs contend and look like a hang).
PYTHONPATH=src python -m pytest tests/qedge -q

# Strict typing + lint gates:
python -m mypy
python -m ruff check src/qedge tests/qedge \
  --select E,F,I,UP,B,ANN,NPY,PD,RUF --ignore ANN101,ANN102 \
  --per-file-ignores "tests/**:ANN"
```

Current status: **mypy --strict clean (31 source files), ruff clean, 231 passed /
2 skipped** (the 2 skips are the live-FMP smoke tests, gated on `FMP_API_KEY` not
being exported to the shell). Run alone, the suite takes ~2.5 min.

What the tests actually prove (audited as rigorous, not vacuous): a planted strong
edge PASSES the gate while same-shape noise FAILS; byte-identical no-lookahead
canaries for features / fracdiff / HMM forward-filter; OOS importance with a
shuffled-y control and an MDI-poison guard; ADWIN/Page-Hinkley flag a real shift
and stay quiet on a stationary stream; cost/partial-fill/capacity monotonicity with
concrete numbers; and the FMP window lower-bound clip.

---

## Extending to intraday / other bar intervals

The daily path is what ships today. Intraday (or any non-daily interval) is a
**deliberate extension**, and much of the scaffolding is already in place. Honest
status of each piece:

**Already present**
- `execution/costs.py` carries an `intraday_latency_ms` knob and a `LatencyModel`
  whose `stated_latency(intraday=True)` returns it.
- The `OptionsNBBOFeed` and `InternalsFeed` Protocols (and the contract's NBBO /
  TICK-TRIN-ADD-VOLD requirements) already exist as typed interfaces — an intraday
  vendor just implements them instead of the `FeedNotAvailable` stubs.
- `startx.data.intraday.get_intraday(...)` already wraps FMP's intraday endpoint
  `historical-chart/{interval}` (1min/5min/…) with `from`/`to` params, and
  `startx.analytics.volume_profile` computes VWAP / POC / value-area / cumulative
  delta from intraday bars.

**What to build (the work you'll need)**
1. **An intraday price feed.** Add `FMPIntradayPriceFeed` in `data/fmp_adapter.py`
   implementing `PriceFeed.history(...)` over `startx.data.intraday.get_intraday`,
   returning bars indexed by an intraday timestamp (not a date). Apply the same
   `asof`/`start` PIT clipping as the daily adapter. (FMP intraday history is
   shorter than daily — confirm the interval's available lookback for your plan.)
2. **Intraday-aware features.** The features in `technical.py` are written against an
   OHLCV frame and are interval-agnostic in form, but their windows are calibrated in
   *days*. Add intraday horizon constants (or scale windows by bars/day) so e.g.
   `rvol`/`mom` mean what you intend on 5-minute bars.
3. **Intraday labeling presets.** Add `intraday` entries alongside `short`/`long` in
   the labeling config with bar-appropriate vertical barriers and `vol_span`, and a
   matching `Horizon` literal in `pipeline.py`.
4. **Real internals / options feeds (optional but contract-relevant).** To honour the
   TICK/TRIN/ADD/VOLD and OPRA-NBBO gates, wire a vendor that has them (e.g.
   Polygon/Databento) behind `InternalsFeed` / `OptionsNBBOFeed`. The pipeline already
   reports them as unpopulated until then.
5. **Latency & impact realism.** Set `execution.intraday_latency_ms` and revisit the
   capacity participation grid — intraday capacity decays far faster than daily.

The validation machinery (CPCV+embargo, DSR/PBO, walk-forward, the no-lookahead
canary, the `TrialLedger`) is interval-agnostic and applies unchanged — but be
extra strict about look-ahead intraday, where end-of-day values leaking into an
intraday feature is the classic trap (the contract calls this out explicitly).

---

## Known findings & caveats

- **No shippable edge found (and that is the correct outcome).** On real
  SPY/QQQ/IWM short-swing over 2018–2026, all three **FAIL** the survival gate
  (DSR 0.70 / 0.67 / 0.32, all < 0.95). A naive trailing-feature + triple-barrier
  model on liquid index ETFs is not a robust edge; adding more features makes the
  *deflated* metric harder to clear, not easier. The scanner's job is to **kill bad
  edges fast** — it does.
- **Cache is keyed by symbol only** (a `startx` property). qedge works around it by
  clipping output to `>= start` and refreshing when the cache doesn't cover the
  requested window, but a longer-term improvement would be a date-aware cache key.
- **Test sizing.** Synthetic pipeline/CLI tests use ~400 bars so the long features
  produce foldable events; run the suite single-threaded.
- **mypy strictness** treats `startx`, pandas, scipy and the ML libs as untyped
  (overrides in `pyproject.toml`) so strict mode enforces qedge's *own* annotations
  without drowning in third-party stub noise; `disallow_untyped_calls` is the one
  strict flag relaxed (qedge legitimately calls untyped `startx`).
- **Secrets:** the FMP key lives only in the gitignored `.env`; it is never written
  into source, config defaults, tests, run records, or commits.
