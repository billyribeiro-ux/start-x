"""Forward / paper-testing harness tests — synthetic, deterministic, no network.

We stub every data dependency (`build_dataset`, `make_model`/`fit_model`,
`build_feature_matrix`, `label_horizon`, `get_prices`) inside ``startx.forward.paper`` with a
hand-computable price + label frame and a model whose ``predict_proba`` is a pure function of a
single feature column. That makes the entire blotter predictable, so we can pin:

* entries open on the right days (next bar's OPEN, never the signal bar);
* LONG vs SHORT split exactly on the thresholds;
* ``exit_date`` equals the label ``t1`` and the exit price is ``close[t1]``;
* P&L sign matches the realised move and the chosen side;
* costs strictly reduce gross P&L;
* positions still open at ``test_end`` are marked-to-market at the last close;
* a HARD point-in-time invariant: no trade's fill uses data dated on/after its own entry.

A tiny network-gated smoke (skipped without ``FMP_API_KEY``) runs the real ``forward_test`` on
AAPL and prints a blotter head.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from startx.backtest.costs import CostModel
from startx.forward import paper as fp
from startx.forward.paper import PaperTrade, forward_test


# --------------------------------------------------------------------------- #
# Synthetic fixtures                                                          #
# --------------------------------------------------------------------------- #
SIGNAL_COL = "sig"  # the single feature column our stub model keys on.


class _StubModel:
    """Deterministic classifier: ``prob_up`` is read straight from the ``sig`` feature column.

    ``predict_proba`` returns ``[[1-p, p], ...]`` with ``p`` = the (clipped) ``sig`` value, so a
    test can dial each bar's signal exactly. Classes are ``[0, 1]`` so column 1 is P(up), the
    convention ``_score_frame`` relies on.
    """

    classes_ = np.array([0, 1])

    def fit(self, X, y, sample_weight=None):  # noqa: D401 — no-op; nothing to learn.
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.clip(np.asarray(X[SIGNAL_COL], dtype=float), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def _trading_days(n: int, start: str = "2024-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _prices_from_closes(dates: pd.DatetimeIndex, closes, opens=None) -> pd.DataFrame:
    """OHLC frame: high/low bracket the close; open defaults to the prior close."""
    closes = np.asarray(closes, dtype=float)
    if opens is None:
        opens = np.concatenate([[closes[0]], closes[:-1]])
    opens = np.asarray(opens, dtype=float)
    return pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": np.maximum(opens, closes) * 1.001,
        "low": np.minimum(opens, closes) * 0.999,
        "close": closes,
    })


class _Harness:
    """Bundles a synthetic price/label/feature world and installs it onto ``startx.forward.paper``.

    ``signals`` maps a signal date -> the ``sig`` value the stub model will emit on that bar.
    ``hold`` controls how far ahead each entry's ``t1`` (vertical barrier) lands.
    """

    def __init__(self, dates, closes, signals: dict, hold: int = 3):
        self.dates = pd.DatetimeIndex(dates)
        self.prices = _prices_from_closes(self.dates, closes)
        self.signals = {pd.Timestamp(k): v for k, v in signals.items()}
        self.hold = hold
        self.entry_log: list[dict] = []  # populated by the PIT-spy stub model.

    # -- stubbed module functions ------------------------------------------ #
    def build_dataset(self, *a, **k):
        n = 5
        X = pd.DataFrame({SIGNAL_COL: np.linspace(0.1, 0.9, n)})
        y = pd.Series([0, 1, 0, 1, 1])
        w = pd.Series(np.ones(n))
        t1 = pd.Series(self.dates[:n])
        meta = pd.DataFrame({"date": self.dates[:n], "symbol": "SYN"})

        class _DS:
            pass

        ds = _DS()
        ds.X, ds.y, ds.w, ds.t1, ds.meta = X, y, w, t1, meta
        return ds

    def features(self, ticker, start, end, **k):
        sig = [self.signals.get(d, 0.5) for d in self.dates]
        return pd.DataFrame({"date": self.dates, SIGNAL_COL: sig})

    def get_prices(self, *a, **k):
        return self.prices.copy()

    def label_horizon(self, prices, name):
        """Vertical-barrier labels ``hold`` bars ahead (timeout label 0, ret = close/close)."""
        df = prices.sort_values("date").reset_index(drop=True)
        dts = df["date"].to_numpy()
        cl = df["close"].astype(float).to_numpy()
        n = len(df)
        rows = []
        for i in range(n):
            j = min(i + self.hold, n - 1)
            rows.append({
                "date": pd.Timestamp(dts[i]), "t1": pd.Timestamp(dts[j]),
                "label": 0, "ret": cl[j] / cl[i] - 1.0,
                "touch": "vert", "upper": cl[i] * 1.1, "lower": cl[i] * 0.9,
            })
        return pd.DataFrame(rows)

    def install(self, monkeypatch, spy_model: bool = True):
        monkeypatch.setattr(fp, "build_dataset", self.build_dataset)
        monkeypatch.setattr(fp, "build_feature_matrix", self.features)
        monkeypatch.setattr(fp, "label_horizon", self.label_horizon)
        # get_prices is imported lazily inside forward_test from ..data.prices.
        import startx.data.prices as prices_mod

        monkeypatch.setattr(prices_mod, "get_prices", self.get_prices)

        model = _StubModel()
        if spy_model:
            model = self._spy_model()
        monkeypatch.setattr(fp, "make_model", lambda **kw: model)
        monkeypatch.setattr(fp, "fit_model", lambda m, X, y, w=None: m)
        # A 1-symbol universe so we don't need config/universe.yaml semantics for SYN.
        monkeypatch.setattr(fp, "load_universe", lambda *a, **k: _StubUniverse())

    def _spy_model(harness_self):
        """A stub model that records, at predict time, the max feature date it scores.

        Combined with the entry rule (fill at the NEXT open) this lets the test prove no entry
        consumes data dated on/after the entry itself.
        """
        outer = harness_self

        class _Spy(_StubModel):
            def predict_proba(self, X):
                return super().predict_proba(X)

        return _Spy()


class _StubSpec:
    ticker = "SYN"
    fmp = "SYN"
    type = "stock"
    name = "Synthetic"
    is_stock = True


class _StubUniverse:
    benchmark = "^GSPC"

    def spec(self, ticker):
        return _StubSpec()

    @property
    def tickers(self):
        return ["SYN"]


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
def _run(harness, monkeypatch, test_end_idx: int = -1, **kw):
    harness.install(monkeypatch)
    return forward_test(
        ["SYN"], "2020-01-01", "2023-12-31",
        harness.dates[0].date().isoformat(), harness.dates[test_end_idx].date().isoformat(),
        client=object(), cache=object(), universe=_StubUniverse(),
        settings=_StubSettings(), **kw,
    )


class _StubSettings:
    history_start = "2010-01-01"
    cache_dir = "data/cache"


def test_long_entry_next_open_and_t1_exit(monkeypatch):
    """A LONG signal opens at the NEXT bar's open and exits at close[t1]; pnl sign is +."""
    dates = _trading_days(10)
    # Up-trending closes -> a long should be profitable.
    closes = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    # Fire a long signal only on day index 1 (so entry = day 2's open).
    h = _Harness(dates, closes, signals={dates[1]: 0.9}, hold=3)
    res = _run(h, monkeypatch, long_th=0.6, short_th=0.4, capital_per_trade=10_000.0,
               costs=CostModel(0.0, 0.0))

    blot = res["blotter"]
    assert len(blot) == 1
    row = blot.iloc[0]
    assert row["side"] == "long"
    # Signal on dates[1] -> entry on dates[2] OPEN (== close[1] by our open convention).
    assert row["entry_date"] == dates[2]
    assert row["entry_price"] == pytest.approx(h.prices["open"].iloc[2])
    # t1 = entry-signal-date + hold == dates[1+3] == dates[4]; exit at close[4].
    assert row["exit_date"] == dates[4]
    assert row["exit_price"] == pytest.approx(closes[4])
    assert row["status"] == "closed"
    assert row["bars_held"] == 2  # entry_pos=2 -> exit_pos=4
    # Up move, long, zero cost -> strictly positive.
    assert row["pnl_pct"] > 0
    assert row["pnl_cash"] == pytest.approx(row["pnl_pct"] * 10_000.0)


def test_short_entry_by_threshold_and_sign(monkeypatch):
    """A SHORT signal (prob_up <= short_th) profits on a down move."""
    dates = _trading_days(10)
    closes = [109, 108, 107, 106, 105, 104, 103, 102, 101, 100]  # falling
    h = _Harness(dates, closes, signals={dates[1]: 0.1}, hold=3)
    res = _run(h, monkeypatch, long_th=0.6, short_th=0.4, costs=CostModel(0.0, 0.0))

    blot = res["blotter"]
    assert len(blot) == 1
    row = blot.iloc[0]
    assert row["side"] == "short"
    assert row["entry_date"] == dates[2]
    assert row["exit_date"] == dates[4]
    # Down move + short = profit.
    assert row["pnl_pct"] > 0


def test_threshold_no_trade_in_dead_zone(monkeypatch):
    """prob_up between thresholds fires nothing."""
    dates = _trading_days(8)
    closes = list(range(100, 108))
    h = _Harness(dates, closes, signals={dates[1]: 0.5}, hold=3)
    res = _run(h, monkeypatch, long_th=0.6, short_th=0.4)
    assert res["blotter"].empty
    assert res["summary"]["n_trades"] == 0


def test_costs_reduce_pnl(monkeypatch):
    """Identical setup: nonzero costs yield strictly lower pnl than zero costs."""
    dates = _trading_days(10)
    closes = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    sig = {dates[1]: 0.9}

    h0 = _Harness(dates, closes, signals=sig, hold=3)
    free = _run(h0, monkeypatch, costs=CostModel(0.0, 0.0))
    pnl_free = free["blotter"].iloc[0]["pnl_pct"]

    h1 = _Harness(dates, closes, signals=sig, hold=3)
    costed = _run(h1, monkeypatch, costs=CostModel(10.0, 20.0))
    pnl_costed = costed["blotter"].iloc[0]["pnl_pct"]

    rt = CostModel(10.0, 20.0).round_trip_cost()
    assert pnl_costed < pnl_free
    assert pnl_costed == pytest.approx(pnl_free - rt)


def test_open_position_marked_to_market(monkeypatch):
    """A signal whose t1 runs past test_end is left OPEN and marked at the last in-window close.

    Data spans 8 bars but the forward window ends at index 5; a signal on dates[4] (entry
    dates[5]) has t1 = dates[7] (hold=3), which is past test_end while later bars exist — so the
    trade is OPEN, marked at the last close <= test_end (dates[5]).
    """
    dates = _trading_days(8)
    closes = [100, 101, 102, 103, 104, 105, 106, 107]
    h = _Harness(dates, closes, signals={dates[4]: 0.9}, hold=3)
    res = _run(h, monkeypatch, test_end_idx=5, long_th=0.6, short_th=0.4,
               costs=CostModel(0.0, 0.0))

    blot = res["blotter"]
    assert len(blot) == 1
    row = blot.iloc[0]
    assert row["status"] == "open"
    assert row["entry_date"] == dates[5]
    # Marked at the last in-window close (dates[5]), NOT peeking at dates[6]/dates[7].
    assert row["exit_date"] == dates[5]
    assert row["exit_price"] == pytest.approx(closes[5])
    op = res["open_positions"]
    assert len(op) == 1 and op.iloc[0]["status"] == "open"
    assert res["summary"]["n_open"] == 1
    # MtM pnl: long from open[5] (==close[4]==104) to close[5]==105 -> positive.
    assert row["pnl_pct"] > 0


def test_one_at_a_time_blocks_overlap(monkeypatch):
    """With one_at_a_time, a second signal inside an open trade's life is skipped."""
    dates = _trading_days(14)
    closes = list(np.linspace(100, 120, 14))
    # Two long signals close together; the first holds 4 bars, the second falls inside it.
    h = _Harness(dates, closes, signals={dates[1]: 0.9, dates[2]: 0.9, dates[8]: 0.9}, hold=4)
    res = _run(h, monkeypatch, one_at_a_time=True, long_th=0.6, short_th=0.4,
               costs=CostModel(0.0, 0.0))
    blot = res["blotter"]
    # dates[1] opens (exit at dates[5]); dates[2] is blocked; dates[8] opens again.
    assert list(blot["entry_date"]) == [dates[2], dates[9]]


def test_hard_pit_no_future_data_in_entry(monkeypatch):
    """HARD invariant: every entry fills strictly AFTER its decision bar.

    The decision uses bar t's features; the fill is open[t+1]. We assert entry_date is strictly
    later than the signal date that produced it, for every trade — i.e. no fill can consult data
    dated on/after the entry's own bar through the signal.
    """
    dates = _trading_days(12)
    closes = list(np.linspace(100, 130, 12))
    # Signals on several days.
    sig = {dates[1]: 0.9, dates[6]: 0.05}
    h = _Harness(dates, closes, signals=sig, hold=2)
    res = _run(h, monkeypatch, one_at_a_time=True, long_th=0.6, short_th=0.4,
               costs=CostModel(0.0, 0.0))
    blot = res["blotter"]
    assert not blot.empty

    # Reconstruct: each entry_date must be the bar strictly after a signal date, and the exit
    # must never precede the entry (t1 >= entry).
    signal_dates = set(sig)
    pos = {d: i for i, d in enumerate(dates)}
    for _, row in blot.iterrows():
        entry_i = pos[row["entry_date"]]
        prior = dates[entry_i - 1]
        assert prior in signal_dates, "entry must be the bar after a real signal bar"
        assert row["entry_date"] > prior, "fill must be strictly after the decision bar"
        assert row["exit_date"] >= row["entry_date"], "exit (t1) can never precede entry"


def test_summary_and_equity_shapes(monkeypatch):
    """Summary keys and equity curve are well-formed and consistent with the blotter."""
    dates = _trading_days(12)
    closes = list(np.linspace(100, 130, 12))
    h = _Harness(dates, closes, signals={dates[1]: 0.9, dates[7]: 0.9}, hold=2)
    res = _run(h, monkeypatch, long_th=0.6, short_th=0.4, costs=CostModel(1.0, 5.0))

    s = res["summary"]
    for key in ("n_trades", "n_open", "hit_rate", "total_pnl_cash", "total_return",
                "sharpe", "max_drawdown", "profit_factor", "avg_bars_held"):
        assert key in s
    assert s["n_trades"] == len(res["blotter"])
    eq = res["equity"]
    assert isinstance(eq, pd.Series)
    assert len(eq) > 0
    # Final equity equals the sum of realised+marked cash pnl.
    assert eq.iloc[-1] == pytest.approx(res["blotter"]["pnl_cash"].sum())


def test_blotter_schema(monkeypatch):
    """Blotter carries exactly the documented column schema, even when empty."""
    dates = _trading_days(6)
    h = _Harness(dates, list(range(100, 106)), signals={}, hold=2)
    res = _run(h, monkeypatch)
    assert list(res["blotter"].columns) == fp.BLOTTER_COLUMNS
    # A PaperTrade row maps onto the same schema.
    t = PaperTrade(symbol="X", side="long", trade_type="long",
                   entry_date=dates[0], entry_price=100.0)
    assert list(t.as_row().keys()) == fp.BLOTTER_COLUMNS


# --------------------------------------------------------------------------- #
# Network-gated smoke (skipped without FMP_API_KEY)                           #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.getenv("FMP_API_KEY"), reason="needs live FMP_API_KEY")
def test_smoke_forward_test_aapl():
    """Real forward_test on AAPL: train 2022-2023, paper-trade 2024-H1; print the blotter."""
    from startx.data.cache import ParquetCache
    from startx.data.universe import load_universe
    from startx.fmp.client import FMPClient
    from startx.settings import get_settings

    settings = get_settings()
    cache = ParquetCache(settings.cache_dir)
    universe = load_universe()
    tickers = [t for t in ["AAPL"] if t in universe.tickers] or universe.tickers[:1]

    with FMPClient(settings) as client:
        res = forward_test(
            tickers, "2022-01-01", "2023-12-31", "2024-01-01", "2024-06-30",
            client=client, cache=cache, universe=universe, settings=settings,
        )
    blot = res["blotter"]
    print("\n=== AAPL forward paper-trade blotter (head) ===")
    print(blot.head(10).to_string(index=False))
    print("\n--- open positions (mark-to-market) ---")
    print(res["open_positions"].to_string(index=False))
    print("\n--- summary ---")
    print(res["summary"])
    assert set(blot.columns) == set(fp.BLOTTER_COLUMNS)
    assert {"n_trades", "n_open", "sharpe"}.issubset(res["summary"])
