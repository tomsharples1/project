# features/label_triple_barrier.py
"""
Triple-barrier labeling (Lopez de Prado) with sensible defaults.

Two modes:
1) Fixed ±% barriers with max holding period (classification: -1/0/+1).
2) Forward return regression over a horizon (no barriers).

If 'high'/'low' are present, barrier hits use intraperiod extremes; otherwise
we fallback to close-to-close paths.

Input formats: same as build_features.py (MultiIndex or flat with 'symbol').

Notes
-----
- No leakage: labels depend only on future path relative to the event time.
- If neither barrier is reached by 'max_holding', the label is 0 (timeout).
- If you prefer asymmetric barriers (e.g., up=1.5*down), set `up_mult` and `down_mult`.

"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple
import numpy as np
import pandas as pd


def _to_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex):
        return df.sort_index()
    assert "symbol" in df.columns, "Expected 'symbol' column when index is not MultiIndex."
    return (
        df
        .set_index(["symbol"], append=True)
        .swaplevel(0, 1)
        .sort_index()
    )


def _compute_daily_vol(close: pd.Series, lookback: int = 50) -> pd.Series:
    """Lopez de Prado daily volatility estimator (EWMA of returns)."""
    # Use abs log returns
    lr = np.log(close).diff().abs()
    vol = lr.ewm(span=lookback, adjust=False).mean()
    return vol


def triple_barrier_labels(
    ohlcv: pd.DataFrame,
    events: Optional[pd.Index] = None,
    up_mult: float = 1.0,
    down_mult: float = 1.0,
    vol_lookback: int = 50,
    max_holding: int = 20,
    min_ret: Optional[float] = None,
) -> pd.DataFrame:
    """
    Assign -1/0/+1 labels using triple barriers.

    Parameters
    ----------
    ohlcv : DataFrame
        OHLCV with (timestamp, symbol) or flat + 'symbol'; must include 'close'.
        'high'/'low' improve barrier detection.
    events : Index, optional
        MultiIndex subset of (timestamp, symbol) indicating event times.
        If None, label every row.
    up_mult, down_mult : float
        Multipliers for the volatility-based vertical barrier heights.
        Upper barrier = close_t * (1 + up_mult * vol_t)
        Lower barrier = close_t * (1 - down_mult * vol_t)
    vol_lookback : int
        Lookback for daily volatility estimator.
    max_holding : int
        Maximum number of bars before timeout.
    min_ret : float, optional
        If set, skip events with |vol_t| < min_ret (filters tiny moves).

    Returns
    -------
    DataFrame with columns:
        - 't1': timestamp of barrier hit/timeout
        - 'label': int in {-1, 0, +1}
        - 'ret': realized return from t to t1 (sign-consistent)
    """
    df = _to_multiindex(ohlcv)
    df = df.sort_index()
    have_hilo = {"high", "low"}.issubset(df.columns)

    out_rows = []

    for sym, g in df.groupby(level=1):
        g = g.sort_index(level=0)
        close = g["close"]
        high = g["high"] if "high" in g else close
        low = g["low"] if "low" in g else close

        vol = _compute_daily_vol(close, vol_lookback)
        if events is None:
            ev_idx = close.index
        else:
            # take only events for this symbol
            ev_idx = events[events.get_level_values(1) == sym]
            ev_idx = ev_idx.intersection(close.index)

        for t in ev_idx:
            c0 = close.loc[t]
            v = vol.loc[t]
            if pd.isna(c0) or pd.isna(v):
                continue
            if (min_ret is not None) and (abs(v) < min_ret):
                continue

            up = c0 * (1.0 + up_mult * v)
            dn = c0 * (1.0 - down_mult * v)

            # forward window
            # note: .get_level_values(0) is timestamp
            ts = close.index.get_level_values(0)
            pos = ts.searchsorted(t[0])  # start position (t included)
            end = min(pos + max_holding, len(close) - 1)

            # Slice forward path EXCLUDING t
            path_idx = close.index[pos + 1 : end + 1]
            if path_idx.empty:
                continue

            path_high = high.loc[path_idx]
            path_low = low.loc[path_idx]

            up_hits = path_high >= up
            dn_hits = path_low <= dn

            hit_time = None
            label = 0

            if up_hits.any() or dn_hits.any():
                # first hit time
                up_time = path_idx[up_hits.argmax()] if up_hits.any() else None
                dn_time = path_idx[dn_hits.argmax()] if dn_hits.any() else None

                if up_time is None:
                    hit_time, label = dn_time, -1
                elif dn_time is None:
                    hit_time, label = up_time, +1
                else:
                    hit_time, label = (up_time, +1) if up_time[0] <= dn_time[0] else (dn_time, -1)
            else:
                # timeout
                hit_time, label = path_idx[-1], 0

            # realized return from t to hit_time using close-to-close
            ret = close.loc[hit_time] / c0 - 1.0
            out_rows.append((t[0], sym, hit_time[0], int(label), float(ret)))

    if not out_rows:
        return pd.DataFrame(columns=["t1", "label", "ret"]).astype({"label": "int64"})

    out = pd.DataFrame(out_rows, columns=["timestamp", "symbol", "t1", "label", "ret"])
    out = out.set_index(["timestamp", "symbol"]).sort_index()
    return out


def forward_return_labels(
    ohlcv: pd.DataFrame,
    horizon: int = 5,
    use_log: bool = False,
) -> pd.Series:
    """
    Simple regression target: forward return over fixed horizon.

    Parameters
    ----------
    ohlcv : DataFrame
        OHLCV with (timestamp, symbol) or flat + 'symbol'.
    horizon : int
        Number of bars ahead to measure the return.
    use_log : bool
        If True, return log-return; else pct return.

    Returns
    -------
    Series indexed by (timestamp, symbol) with 'y' values.
    """
    df = _to_multiindex(ohlcv).sort_index()
    y_list = []
    for sym, g in df.groupby(level=1):
        close = g["close"]
        if use_log:
            y = (np.log(close.shift(-horizon)) - np.log(close)).rename("y")
        else:
            y = (close.shift(-horizon) / close - 1.0).rename("y")
        y_list.append(y)
    y = pd.concat(y_list).sort_index()
    return y


if __name__ == "__main__":
    # Synthetic quick test
    idx = pd.date_range("2023-01-01", periods=200, freq="D")
    syms = ["AAA", "BBB"]
    rng = np.random.default_rng(42)
    rows = []
    for s in syms:
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
        hi = px * (1 + np.abs(rng.normal(0.002, 0.01, len(idx))))
        lo = px * (1 - np.abs(rng.normal(0.002, 0.01, len(idx))))
        vol = rng.integers(1e5, 2e5, len(idx))
        rows.append(pd.DataFrame(
            {"open": px, "high": hi, "low": lo, "close": px, "volume": vol},
            index=pd.MultiIndex.from_product([[s], idx], names=["symbol", "timestamp"])
        ))
    data = pd.concat(rows).swaplevel(0, 1).sort_index()

    labels = triple_barrier_labels(data, up_mult=1.0, down_mult=1.0, max_holding=10)
    print(labels.head())

    y = forward_return_labels(data, horizon=5)
    print(y.head())
