"""López de Prado triple-barrier labeling for LONG & SHORT swing signals.

For each entry date ``t`` (entry assumed at ``close[t]``) we place three barriers:

* an **upper** profit-taking barrier at ``close[t] * (1 + pt_mult * vol[t])``,
* a **lower** stop-loss barrier at ``close[t] * (1 - sl_mult * vol[t])``,
* a **vertical** (time) barrier ``horizon_days`` trading rows ahead.

We walk forward bar-by-bar and detect the *first touch* using each bar's HIGH (vs the
upper barrier) and LOW (vs the lower barrier). The first barrier touched determines the
label: ``+1`` (upper / profit), ``-1`` (lower / stop), ``0`` (vertical / timeout).

The emitted ``t1`` is the exact timestamp of first touch (or the vertical-barrier date).
``t1`` drives purging in leakage-proof cross-validation, so it must be exact.

Conservative same-bar tie-break: if a single bar's HIGH reaches ``upper`` *and* its LOW
reaches ``lower`` (we cannot know intrabar ordering from OHLC alone), we resolve toward
the stop — label ``-1``, ``touch == 'sl'``. This is the pessimistic assumption and avoids
optimistically over-counting wins.

See *Advances in Financial Machine Learning*, López de Prado, Ch. 3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_COLUMNS = ["date", "t1", "label", "ret", "touch", "upper", "lower"]


def daily_vol(close: pd.Series, span: int = 63) -> pd.Series:
    """EWMA standard deviation of daily close-to-close returns (barrier scale).

    Returns a series aligned to ``close``'s index. The first observation has no prior
    return and is back-filled from the first finite estimate so barriers are never NaN.
    """
    close = close.astype(float)
    ret = close.pct_change()
    vol = ret.ewm(span=span, min_periods=1).std()
    # First row's std is NaN (single obs); back-fill so every entry has a usable scale.
    return vol.bfill()


def triple_barrier_labels(
    prices: pd.DataFrame,
    *,
    horizon_days: int,
    pt_mult: float,
    sl_mult: float,
    vol_span: int = 63,
    vol: pd.Series | None = None,
) -> pd.DataFrame:
    """Label every entry bar with the triple-barrier method (LONG-side convention).

    Parameters
    ----------
    prices:
        DataFrame with at least ``date``, ``high``, ``low``, ``close`` columns
        (schema of :func:`startx.data.prices.get_prices`). Assumed chronologically
        sorted; it is defensively re-sorted by ``date``.
    horizon_days:
        Number of trading rows ahead for the vertical barrier (``t1`` at row ``i +
        horizon_days``). Must be a positive integer.
    pt_mult, sl_mult:
        Profit-take / stop-loss barrier widths as multiples of ``vol[t]``.
    vol_span:
        EWMA span for :func:`daily_vol` when ``vol`` is not supplied.
    vol:
        Optional precomputed volatility series (positionally aligned to ``prices``).
        When ``None`` it is computed from ``close`` via :func:`daily_vol`.

    Returns
    -------
    DataFrame indexed by integer position (one row per entry ``t``) with columns
    ``date, t1, label, ret, touch, upper, lower`` (see :data:`LABEL_COLUMNS`).
    """
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")
    if prices.empty:
        return pd.DataFrame(columns=LABEL_COLUMNS)

    df = prices.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"]).to_numpy()
    close = df["close"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    n = len(df)

    if vol is None:
        vol_arr = daily_vol(df["close"], span=vol_span).to_numpy()
    else:
        vol_arr = np.asarray(vol, dtype=float)
        if len(vol_arr) != n:
            raise ValueError(
                f"vol length {len(vol_arr)} != prices length {n}; must be aligned"
            )

    out_date: list[pd.Timestamp] = []
    out_t1: list[pd.Timestamp] = []
    out_label: list[int] = []
    out_ret: list[float] = []
    out_touch: list[str] = []
    out_upper: list[float] = []
    out_lower: list[float] = []

    for i in range(n):
        c0 = close[i]
        v = vol_arr[i]
        upper = c0 * (1.0 + pt_mult * v)
        lower = c0 * (1.0 - sl_mult * v)

        # Vertical-barrier row: i + horizon_days, clamped to the last available row.
        vert_idx = i + horizon_days
        truncated = vert_idx >= n
        last_idx = min(vert_idx, n - 1)

        label = 0
        touch = "vert"
        t1_idx = last_idx

        # Walk forward from the first bar AFTER entry through the vertical barrier.
        for j in range(i + 1, last_idx + 1):
            hit_pt = high[j] >= upper
            hit_sl = low[j] <= lower
            if hit_pt and hit_sl:
                # Ambiguous intrabar ordering -> resolve conservatively toward the stop.
                label, touch, t1_idx = -1, "sl", j
                break
            if hit_sl:
                label, touch, t1_idx = -1, "sl", j
                break
            if hit_pt:
                label, touch, t1_idx = 1, "pt", j
                break

        # No barrier hit: vertical/timeout. If the horizon was truncated by end-of-data,
        # t1 is the last available date and the label remains 0 (neither barrier reached).
        if touch == "vert" and truncated:
            touch = "vert"  # explicit: timeout at last available date

        out_date.append(pd.Timestamp(dates[i]))
        out_t1.append(pd.Timestamp(dates[t1_idx]))
        out_label.append(int(label))
        out_ret.append(float(close[t1_idx] / c0 - 1.0))
        out_touch.append(touch)
        out_upper.append(float(upper))
        out_lower.append(float(lower))

    result = pd.DataFrame(
        {
            "date": out_date,
            "t1": out_t1,
            "label": out_label,
            "ret": out_ret,
            "touch": out_touch,
            "upper": out_upper,
            "lower": out_lower,
        }
    )
    return result.reset_index(drop=True)
