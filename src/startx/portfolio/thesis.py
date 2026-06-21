"""Human-judgment layer — give every trade a PM's read: regime, conviction, and a thesis.

This is the "highest and most human level" of the book. The mechanical sleeves (breakout, IBS dip,
VRP/VVIX fear longs) decide *when* to enter; this module decides *what to think about it*. For each
trade it fuses three things an elite portfolio manager fuses in their head:

1. **Regime** — what kind of market is this? A compact, point-in-time read at the entry date built
   from VIX vs its adaptive (zero-drift) neutral, VVIX (vol-of-vol) percentile, the trend (close vs
   SMA200), and realized vol. "fear: VIX 1.4x neutral, VVIX p93, uptrend" vs "calm: VIX 0.9x
   neutral, trend intact".

2. **Conviction** — a 1-3 score (low / medium / high) from (a) a *sleeve durability prior* — the
   breakout sleeve is the only one with real drift-adjusted alpha out-of-sample, so it starts
   strongest; the fear sleeves (vrp/vvix) are a medium fear-premium harvested small; IBS is a
   medium-low exposure-timing edge — plus (b) *regime alignment* (fear sleeves earn a bump only
   when VIX is well above neutral AND VVIX is extreme; breakout earns a bump in a clean low-vol
   uptrend), plus (c) an *optional* meta-model P(win) used as a soft tilt, never a hard gate (the
   cross-desk verdict: the meta-filter is fragile and only smooths drawdown).

3. **Thesis** — one human sentence a PM would actually say, weaving in the *causal why* when the
   causal decoder (``startx.events.analyze_symbol``) attributes a catalyst near the entry ("...this
   dip was a macro-driven flush, now snapping back"), and naming the nearest known risk ahead
   (e.g. an FOMC two days out) when the macro calendar is reachable.

The whole module degrades gracefully: if FMP / the causal decoder is slow or unavailable, the
thesis still reads cleanly from the price-and-vol regime alone. The only hard dependency is the
``spy`` and ``aux`` price frames you already have in cache.

Public surface
--------------
``annotate_theses(ledger, spy, aux, *, client=None, cache=None) -> ledger``
    Fill the empty ``regime``, ``conviction``, ``thesis`` columns of a portfolio ledger.
``summarize_book(ledger) -> str``
    One human paragraph describing what the book is doing now.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..strategy.vix_capitulation import adaptive_neutral

# ---------------------------------------------------------------------------
# Sleeve taxonomy — the durability prior. This is a *belief about the edge*, set from FINDINGS.md /
# the cross-desk verdict, not from any single backtest's in-sample P&L.
# ---------------------------------------------------------------------------

# Canonical family for each sleeve name the engine might emit. We match loosely (substring) so
# "breakout", "momentum_breakout", "ibs_dip", "vrp_high", "vvix_spike", "vix_capitulation" all land.
_SLEEVE_FAMILY: dict[str, str] = {
    "breakout": "breakout",
    "momentum": "breakout",
    "ibs": "ibs",
    "meanrev": "ibs",
    "mean_reversion": "ibs",
    "rsi": "ibs",
    "dip": "ibs",
    "vrp": "fear",
    "vvix": "fear",
    "vix_cap": "fear",
    "capitulation": "fear",
}

# Base conviction prior per family (continuous 0-1; discretized to 1/2/3 at the end). Breakout is the
# only sleeve with genuine drift-adjusted OOS alpha (PF 3.64 TEST) -> highest base. The fear family
# is a real but thin risk premium, sized small -> medium. IBS is exposure timing, a positive but
# small edge -> medium-low.
_FAMILY_PRIOR: dict[str, float] = {
    "breakout": 0.62,
    "fear": 0.50,
    "ibs": 0.42,
    "unknown": 0.45,
}

_FAMILY_BLURB: dict[str, str] = {
    "breakout": "the 20-day breakout (trend sleeve, the book's primary edge)",
    "fear": "a fear-premium long (selling insurance into panic, sized small)",
    "ibs": "an oversold-dip buy (exposure timing in an uptrend)",
    "unknown": "this entry",
}

_CONV_LABEL = {1: "low", 2: "medium", 3: "high"}

# VVIX percentile thresholds for the regime read / conviction (point-in-time, trailing 1y).
_VVIX_HOT = 0.90    # >= this -> "extreme" vol-of-vol, the fear sleeves' green light
_VVIX_WARM = 0.75
_VIX_FEAR_RATIO = 1.25   # VIX this many x its neutral -> "fear"
_VIX_STRESS_RATIO = 1.6  # well above neutral -> deep stress


def _family_of(sleeve: str) -> str:
    """Map a raw sleeve label to its durability family (loose substring match)."""
    s = str(sleeve).lower()
    for key, fam in _SLEEVE_FAMILY.items():
        if key in s:
            return fam
    return "unknown"


# ---------------------------------------------------------------------------
# Regime state — the point-in-time market read at an entry date.
# ---------------------------------------------------------------------------

@dataclass
class RegimeState:
    """A compact, point-in-time read of the market at one entry date.

    All fields are computed from data available *at or before* ``date`` (no lookahead): the VVIX
    percentile uses a trailing 1y window, the VIX neutral is the rolling zero-drift attractor, and
    the trend/realized-vol come from the close history up to ``date``.
    """

    date: pd.Timestamp
    vix: float = float("nan")
    neutral: float = float("nan")
    vix_ratio: float = float("nan")     # vix / neutral
    vvix: float = float("nan")
    vvix_pct: float = float("nan")      # trailing-1y percentile of VVIX, in [0,1]
    term_slope: float = float("nan")    # VIX3M / VIX  (>1 = contango/calm, <1 = backwardation/stress)
    above_sma200: bool | None = None
    sma_gap_pct: float = float("nan")   # close vs SMA200, in %
    realized_vol: float = float("nan")  # 20d annualized, VIX-comparable points
    label: str = "neutral"              # "fear" | "stress" | "calm" | "neutral"

    @property
    def trend_word(self) -> str:
        if self.above_sma200 is None:
            return "trend unknown"
        return "uptrend" if self.above_sma200 else "downtrend"

    def describe(self) -> str:
        """Render the compact regime string stored in the ledger."""
        bits: list[str] = []
        if np.isfinite(self.vix_ratio):
            bits.append(f"VIX {self.vix_ratio:.2g}x neutral")
        if np.isfinite(self.vvix_pct):
            bits.append(f"VVIX p{int(round(self.vvix_pct * 100))}")
        bits.append(self.trend_word)
        if np.isfinite(self.term_slope) and self.term_slope < 1.0:
            bits.append("term inverted")
        return f"{self.label}: " + ", ".join(bits)


def _pct_rank(series: pd.Series, value: float, window: int = 252) -> float:
    """Trailing-``window`` percentile of ``value`` within ``series`` (the tail up to & incl. now)."""
    tail = series.dropna().tail(window)
    if tail.empty or not np.isfinite(value):
        return float("nan")
    return float((tail <= value).mean())


def _as_close_series(df: pd.DataFrame | None) -> pd.Series | None:
    """Coerce a cached OHLC frame to a date-indexed close series (sorted, deduped)."""
    if df is None or len(df) == 0 or "close" not in df.columns or "date" not in df.columns:
        return None
    s = df.copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date").drop_duplicates("date").set_index("date")["close"].astype(float)
    return s


@dataclass
class _RegimeModel:
    """Precomputed, point-in-time regime inputs aligned to the SPY calendar.

    Built once per ``annotate_theses`` call from the SPY + aux frames so each trade is an O(1)
    lookup. Everything here is causal: SMA200, realized vol, the VIX neutral and the VVIX percentile
    are all trailing as of each date.
    """

    spy_close: pd.Series
    sma200: pd.Series
    realized: pd.Series
    vix: pd.Series | None = None
    neutral: pd.Series | None = None
    vvix: pd.Series | None = None
    vix3m: pd.Series | None = None
    vvix_raw: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @classmethod
    def build(cls, spy: pd.DataFrame, aux: Mapping[str, pd.DataFrame] | None) -> "_RegimeModel":
        close = _as_close_series(spy)
        if close is None or close.empty:
            raise ValueError("annotate_theses needs a SPY frame with 'date' and 'close' columns.")
        aux = aux or {}

        def pick(*names: str) -> pd.Series | None:
            for n in names:
                if n in aux:
                    s = _as_close_series(aux[n])
                    if s is not None and not s.empty:
                        return s
            return None

        vix = pick("_VIX", "VIX", "^VIX", "vix")
        vvix = pick("_VVIX", "VVIX", "^VVIX", "vvix")
        vix3m = pick("_VIX3M", "VIX3M", "^VIX3M", "vix3m")

        sma200 = close.rolling(200, min_periods=50).mean()
        realized = close.pct_change().rolling(20).std() * np.sqrt(252) * 100.0

        neutral = None
        if vix is not None:
            v = vix.reset_index().rename(columns={vix.name or "close": "close"})
            v.columns = ["date", "close"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                neutral = pd.Series(adaptive_neutral(v["close"]).to_numpy(), index=vix.index)

        return cls(
            spy_close=close, sma200=sma200, realized=realized, vix=vix, neutral=neutral,
            vvix=vvix, vix3m=vix3m, vvix_raw=vvix if vvix is not None else pd.Series(dtype=float),
        )

    @staticmethod
    def _asof(series: pd.Series | None, when: pd.Timestamp) -> float:
        if series is None or series.empty:
            return float("nan")
        sub = series.loc[:when]
        return float(sub.iloc[-1]) if len(sub) else float("nan")

    def at(self, when: pd.Timestamp) -> RegimeState:
        """Point-in-time regime state as of ``when`` (the entry date)."""
        when = pd.Timestamp(when).normalize()
        st = RegimeState(date=when)

        close = self._asof(self.spy_close, when)
        sma = self._asof(self.sma200, when)
        if np.isfinite(close) and np.isfinite(sma):
            st.above_sma200 = bool(close > sma)
            st.sma_gap_pct = (close / sma - 1.0) * 100.0
        st.realized_vol = self._asof(self.realized, when)

        st.vix = self._asof(self.vix, when)
        st.neutral = self._asof(self.neutral, when)
        if np.isfinite(st.vix) and np.isfinite(st.neutral) and st.neutral > 0:
            st.vix_ratio = st.vix / st.neutral

        st.vvix = self._asof(self.vvix, when)
        if self.vvix is not None and np.isfinite(st.vvix):
            st.vvix_pct = _pct_rank(self.vvix.loc[:when], st.vvix)

        v3m = self._asof(self.vix3m, when)
        if np.isfinite(v3m) and np.isfinite(st.vix) and st.vix > 0:
            st.term_slope = v3m / st.vix

        st.label = self._classify(st)
        return st

    @staticmethod
    def _classify(st: RegimeState) -> str:
        r = st.vix_ratio
        if np.isfinite(r):
            if r >= _VIX_STRESS_RATIO:
                return "stress"
            if r >= _VIX_FEAR_RATIO:
                return "fear"
            if r <= 0.95:
                return "calm"
            return "neutral"
        # No VIX -> fall back to realized vol bands (VIX-comparable points).
        rv = st.realized_vol
        if np.isfinite(rv):
            if rv >= 28:
                return "stress"
            if rv >= 20:
                return "fear"
            if rv <= 12:
                return "calm"
        return "neutral"


# ---------------------------------------------------------------------------
# Conviction — fuse the durability prior, regime alignment, and the optional meta tilt.
# ---------------------------------------------------------------------------

def _alignment_bonus(family: str, st: RegimeState) -> tuple[float, str]:
    """How well does the regime fit this sleeve's edge? Returns (delta in [-.2,.2], reason)."""
    if family == "breakout":
        # Breakout wants a clean, calm uptrend with quiet vol-of-vol. It hates fear/stress.
        if st.above_sma200 and st.label == "calm" and (
            not np.isfinite(st.vvix_pct) or st.vvix_pct <= _VVIX_WARM
        ):
            return 0.18, "clean low-vol uptrend confirms the trend"
        if st.above_sma200 is False:
            return -0.20, "below the 200-day — a breakout against the trend is fragile"
        if st.label in ("fear", "stress"):
            return -0.12, "buying strength into a fear spike is lower-quality"
        return 0.05, "trend intact"

    if family == "fear":
        # Fear sleeves are paid to step in when fear is both elevated AND extreme in vol-of-vol.
        hot_vvix = np.isfinite(st.vvix_pct) and st.vvix_pct >= _VVIX_HOT
        deep_vix = np.isfinite(st.vix_ratio) and st.vix_ratio >= _VIX_FEAR_RATIO
        if hot_vvix and deep_vix:
            return 0.20, "VIX well above neutral and VVIX extreme — fear is richly over-paid"
        if hot_vvix or deep_vix:
            return 0.08, "fear is elevated, the premium is there"
        return -0.10, "fear is muted — thin premium, little edge to harvest"

    if family == "ibs":
        # A dip buy wants an intact uptrend; a calm tape is a cleaner bounce setup than a panic.
        if st.above_sma200:
            base = 0.08
            if st.label in ("calm", "neutral"):
                return base + 0.04, "oversold close in an intact uptrend"
            return base, "oversold dip, uptrend holding through the vol"
        return -0.18, "dip-buying below the 200-day fights the trend"

    return 0.0, ""


def _meta_tilt(p_win: float | None) -> tuple[float, str]:
    """Optional meta-model P(win) as a *soft* tilt (never a gate). Centered at 0.5."""
    if p_win is None or not np.isfinite(p_win):
        return 0.0, ""
    delta = float(np.clip((p_win - 0.5) * 0.30, -0.10, 0.10))  # +-0.10 at the extremes
    if p_win >= 0.6:
        return delta, f"meta-model leans favorable (P_win {p_win:.0%})"
    if p_win <= 0.4:
        return delta, f"meta-model is cautious (P_win {p_win:.0%})"
    return delta, ""


@dataclass
class Conviction:
    score: int          # 1, 2, or 3
    label: str          # "low" | "medium" | "high"
    raw: float          # continuous 0-1 before discretizing (for transparency / sorting)
    reasons: list[str] = field(default_factory=list)

    def render(self) -> str:
        return f'{self.score} "{self.label}"'


def _score_conviction(family: str, st: RegimeState, p_win: float | None) -> Conviction:
    base = _FAMILY_PRIOR.get(family, _FAMILY_PRIOR["unknown"])
    reasons: list[str] = []
    align, why = _alignment_bonus(family, st)
    if why:
        reasons.append(why)
    tilt, twhy = _meta_tilt(p_win)
    if twhy:
        reasons.append(twhy)
    raw = float(np.clip(base + align + tilt, 0.0, 1.0))
    # Discretize: a sleeve with a strong prior AND aligned regime reaches 3; a weak/misaligned one
    # drops to 1. The breakpoints are chosen so the breakout prior (0.62) lands at 3 only when the
    # regime confirms it, and any sleeve falls to 1 when the regime actively fights it.
    if raw >= 0.66:
        score = 3
    elif raw >= 0.46:
        score = 2
    else:
        score = 1
    return Conviction(score=score, label=_CONV_LABEL[score], raw=round(raw, 3), reasons=reasons)


# ---------------------------------------------------------------------------
# Causal enrichment — pull the "why" from the events decoder, degrade gracefully.
# ---------------------------------------------------------------------------

# Plain-English glosses for the decoder's cause types, in PM voice.
_CAUSE_GLOSS: dict[str, str] = {
    "macro": "a macro-driven",
    "earnings": "an earnings-driven",
    "news": "a headline-driven",
    "gap_volume": "a gap-and-volume",
    "analyst_grade": "an analyst-rating",
    "price_target": "a price-target",
    "insider": "an insider-flow",
    "congress": "a congressional-flow",
}


@dataclass
class CausalRead:
    """The decoder's attribution for the move nearest an entry (within a few sessions)."""

    direction: str          # "up" | "down"
    cause_type: str
    label: str              # raw decoder label, e.g. "Macro: FOMC Minutes"
    confidence: float
    days_before_entry: int  # how many sessions before the entry the move happened (>=0)

    def phrase(self, family: str) -> str:
        """A clause to weave into the thesis, e.g. 'this dip was a macro-driven flush, now ...'."""
        gloss = _CAUSE_GLOSS.get(self.cause_type, "a")
        if self.direction == "down":
            move = "flush"
            tail = "now snapping back" if family in ("ibs", "fear") else "and momentum is repairing"
            return f"this dip was {gloss} {move}, {tail}"
        move = "thrust"
        return f"this leg up was {gloss} {move}, and it is following through"


def _causal_read(
    symbol: str, entry: pd.Timestamp, *, client: Any, cache: Any,
    lookback_sessions: int = 6,
) -> CausalRead | None:
    """Ask the causal decoder for the dominant attributed move just before ``entry``.

    Looks back a short window (so the cause is plausibly *this* trade's reason) and returns the
    highest-confidence event. Returns ``None`` on any failure or if FMP is unavailable — the thesis
    must still work without it.
    """
    if client is None:
        return None
    try:
        from ..events.engine import analyze_symbol
    except Exception:
        return None
    entry = pd.Timestamp(entry).normalize()
    start = (entry - timedelta(days=lookback_sessions * 2 + 4)).date().isoformat()
    end = entry.date().isoformat()
    try:
        res = analyze_symbol(
            symbol, start, end, client=client, cache=cache, ar_threshold=2.0, lookback_days=2,
        )
    except Exception:
        return None
    ev = res.events
    if ev is None or ev.empty:
        return None
    ev = ev.copy()
    ev["date"] = pd.to_datetime(ev["date"])
    ev = ev[ev["date"] <= entry]
    if ev.empty:
        return None
    ev["days_before"] = (entry - ev["date"]).dt.days
    ev = ev[ev["days_before"] <= lookback_sessions + 2]
    if ev.empty:
        return None
    # Prefer the highest-confidence attributed move in the window.
    row = ev.sort_values(["confidence", "abs_ar_pct"], ascending=False).iloc[0]
    ctype = row.get("top_cause_type") or "news"
    if not isinstance(ctype, str):
        ctype = "news"
    return CausalRead(
        direction=str(row["direction"]),
        cause_type=ctype,
        label=str(row.get("top_cause") or "Unattributed"),
        confidence=float(row.get("confidence") or 0.0),
        days_before_entry=int(row["days_before"]),
    )


def _upcoming_risk(
    entry: pd.Timestamp, *, client: Any, cache: Any, horizon_days: int = 5,
) -> str | None:
    """Name the nearest high-impact macro event in the days *after* an entry, if reachable.

    Used only to color the thesis ('...FOMC two days out is the only risk'). Best-effort: returns
    ``None`` if FMP is down or nothing notable is on the calendar.
    """
    if client is None:
        return None
    try:
        from ..fmp import endpoints as ep
    except Exception:
        return None
    entry = pd.Timestamp(entry).normalize()
    nxt = (entry + timedelta(days=1)).date().isoformat()
    end = (entry + timedelta(days=horizon_days)).date().isoformat()
    try:
        cal = ep.economic_calendar(client, nxt, end)
    except Exception:
        return None
    if cal is None or cal.empty or "event" not in cal.columns or "ts" not in cal.columns:
        return None
    cal = cal.copy()
    cal["ts"] = pd.to_datetime(cal["ts"])
    if "impact" in cal.columns:
        cal = cal[cal["impact"].astype(str).str.lower() == "high"]
    if "country" in cal.columns:
        us = cal[cal["country"].astype(str).isin(["US", "USA"])]
        cal = us if not us.empty else cal
    cal = cal[cal["ts"] > entry].sort_values("ts")
    if cal.empty:
        return None
    row = cal.iloc[0]
    days = max(1, int((pd.Timestamp(row["ts"]).normalize() - entry).days))
    when = "tomorrow" if days == 1 else f"{days} days out"
    return f"{str(row['event'])} {when}"


# ---------------------------------------------------------------------------
# Thesis composition — one PM sentence.
# ---------------------------------------------------------------------------

def _vol_word(st: RegimeState) -> str:
    if st.label == "stress":
        return "vol is screaming"
    if st.label == "fear":
        return "vol is bid"
    if st.label == "calm":
        return "vol is calm"
    if np.isfinite(st.realized_vol):
        return "vol is middling"
    return "vol read unavailable"


def _compose_thesis(
    family: str, sleeve: str, st: RegimeState, conv: Conviction,
    causal: CausalRead | None, risk: str | None,
) -> str:
    """Assemble the single human sentence."""
    blurb = _FAMILY_BLURB.get(family, _FAMILY_BLURB["unknown"])

    if family == "breakout":
        head = f"Buy {blurb} — {st.trend_word} intact, {_vol_word(st)}, momentum confirming"
    elif family == "fear":
        head = f"Step into {blurb} — {_vol_word(st)}"
        if np.isfinite(st.vix_ratio):
            head += f" at {st.vix_ratio:.2g}x neutral"
        if np.isfinite(st.vvix_pct):
            head += f", VVIX p{int(round(st.vvix_pct * 100))}"
        head += "; the snap-back is loading"
    elif family == "ibs":
        head = f"Buy {blurb} — oversold close, {st.trend_word} holding, {_vol_word(st)}"
    else:
        head = f"Take {blurb} — {st.trend_word}, {_vol_word(st)}"

    clauses: list[str] = [head]
    if causal is not None and causal.confidence >= 0.15:
        clauses.append(causal.phrase(family))
    if risk:
        clauses.append(f"{risk} is the watch-item")
    elif family == "breakout" and st.label in ("fear", "stress"):
        clauses.append("the live risk is buying strength into a vol spike")

    sentence = "; ".join(clauses)
    tag = f" [{conv.label} conviction]"
    if not sentence.endswith("."):
        sentence += "."
    return sentence + tag


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------

def _find_meta_pwin(row: pd.Series) -> float | None:
    """Pull an optional meta-model P(win) from whatever column the engine may have used."""
    for col in ("p_win", "pwin", "meta_p", "prob_win", "meta_pwin"):
        if col in row.index:
            val = row[col]
            if val is not None and np.isfinite(val):
                return float(val)
    return None


def annotate_theses(
    ledger: pd.DataFrame,
    spy: pd.DataFrame,
    aux: Mapping[str, pd.DataFrame] | None = None,
    *,
    client: Any = None,
    cache: Any = None,
    symbol: str = "SPY",
    use_causal: bool = True,
) -> pd.DataFrame:
    """Fill the ``regime``, ``conviction`` and ``thesis`` columns of a portfolio ledger.

    Parameters
    ----------
    ledger
        One row per trade. Must carry at least ``sleeve`` and ``entry_date``; ``entry_price``,
        ``exit_date``, ``ret``, ``outcome`` are used when present. The three target columns are
        created if absent and overwritten if present-but-empty.
    spy
        SPY OHLC frame (``date``, ``close``, ...). Drives trend, SMA200 and realized vol.
    aux
        Mapping of context series by cache name, e.g. ``{"_VIX": df, "_VVIX": df, "_VIX3M": df}``.
        Missing series degrade gracefully (regime falls back to realized vol).
    client, cache
        Optional FMP client + Parquet cache. When given (and ``use_causal``), the causal decoder
        enriches each thesis with the attributed "why" and the macro calendar names the next risk.
        When ``None`` or unreachable, the thesis is built from the price/vol regime alone.

    Returns
    -------
    The same ledger (a copy) with ``regime``, ``conviction``, ``thesis`` filled.
    """
    out = ledger.copy()
    for col in ("regime", "conviction", "thesis"):
        if col not in out.columns:
            out[col] = pd.Series([pd.NA] * len(out), dtype=object)
        else:
            out[col] = out[col].astype(object)
    if out.empty:
        return out

    model = _RegimeModel.build(spy, aux)

    # Causal/calendar lookups hit the network; cache per entry date so duplicate dates pay once.
    causal_cache: dict[pd.Timestamp, CausalRead | None] = {}
    risk_cache: dict[pd.Timestamp, str | None] = {}
    causal_live = bool(use_causal and client is not None)

    regimes: list[str] = []
    convictions: list[str] = []
    theses: list[str] = []

    for _, row in out.iterrows():
        sleeve = row.get("sleeve", "unknown")
        family = _family_of(sleeve)
        entry = pd.Timestamp(row["entry_date"]).normalize()

        st = model.at(entry)
        conv = _score_conviction(family, st, _find_meta_pwin(row))

        causal: CausalRead | None = None
        risk: str | None = None
        if causal_live:
            if entry not in causal_cache:
                causal_cache[entry] = _causal_read(
                    symbol, entry, client=client, cache=cache,
                )
            causal = causal_cache[entry]
            if entry not in risk_cache:
                risk_cache[entry] = _upcoming_risk(entry, client=client, cache=cache)
            risk = risk_cache[entry]

        regimes.append(st.describe())
        convictions.append(conv.render())
        theses.append(_compose_thesis(family, sleeve, st, conv, causal, risk))

    out["regime"] = regimes
    out["conviction"] = convictions
    out["thesis"] = theses
    return out


def _conviction_score(val: Any) -> int | None:
    """Parse the leading integer out of a rendered conviction cell ('3 \"high\"')."""
    try:
        return int(str(val).strip().split()[0])
    except (ValueError, IndexError):
        return None


def summarize_book(ledger: pd.DataFrame) -> str:
    """A short human paragraph: what is the book doing now, net regime, aggregate conviction.

    Reads the annotated ledger (or annotates nothing if the columns are empty) and describes the
    live posture the way a PM would brief the desk in two or three sentences.
    """
    if ledger is None or ledger.empty:
        return "The book is flat — no open or recent trades to read."

    df = ledger.copy()
    # Sleeve mix.
    fam_counts: dict[str, int] = {}
    for s in df.get("sleeve", pd.Series(dtype=object)):
        fam_counts[_family_of(s)] = fam_counts.get(_family_of(s), 0) + 1
    fam_counts.pop("unknown", None)
    n = len(df)

    fam_names = {
        "breakout": "trend/breakout", "fear": "fear-premium", "ibs": "dip-buy",
    }
    mix = ", ".join(
        f"{cnt} {fam_names.get(f, f)}" for f, cnt in sorted(fam_counts.items(), key=lambda kv: -kv[1])
    ) or "mixed"

    # Net regime: most common regime label among the trades.
    reg_label = "mixed"
    if "regime" in df.columns and df["regime"].notna().any():
        labels = (
            df["regime"].dropna().astype(str).str.split(":").str[0].str.strip()
        )
        if not labels.empty:
            reg_label = labels.mode().iloc[0]

    # Aggregate conviction.
    conv_word = "mixed"
    avg_conv = float("nan")
    if "conviction" in df.columns and df["conviction"].notna().any():
        scores = [s for s in (_conviction_score(v) for v in df["conviction"]) if s is not None]
        if scores:
            avg_conv = float(np.mean(scores))
            conv_word = ("high" if avg_conv >= 2.5 else "medium" if avg_conv >= 1.75 else "low")

    # Win rate, if outcomes are present (descriptive, not predictive).
    perf = ""
    if "outcome" in df.columns and df["outcome"].notna().any():
        o = df["outcome"].astype(str).str.upper()
        wins = o.isin(["WIN", "SCRATCH"]).sum()
        perf = f" Realized so far: {wins}/{len(o)} non-losers."

    regime_gloss = {
        "fear": "leaning into fear (vol bid above neutral)",
        "stress": "leaning hard into stress (vol screaming)",
        "calm": "riding a calm, low-vol tape",
        "neutral": "in a neutral-vol tape",
        "mixed": "spanning mixed regimes",
    }.get(reg_label, f"in a {reg_label} tape")

    conv_clause = (
        f"aggregate conviction is {conv_word}"
        + (f" (avg {avg_conv:.1f}/3)" if np.isfinite(avg_conv) else "")
    )

    return (
        f"The book is running {n} position(s) — {mix} — {regime_gloss}; {conv_clause}. "
        f"Net posture: the desk is {regime_gloss}, sizing the fear sleeves small and letting the "
        f"trend sleeve carry the weight where the regime confirms it.{perf}"
    )


__all__ = [
    "RegimeState",
    "Conviction",
    "CausalRead",
    "annotate_theses",
    "summarize_book",
]
