# features/build_features.py
"""
Feature engineering utilities for market data.

Supported input formats
-----------------------
1) MultiIndex DataFrame indexed by (timestamp, symbol) with columns:
   ['open','high','low','close','volume']
2) Flat DataFrame indexed by timestamp with a 'symbol' column and the same OHLCV columns.

All rolling features are shifted by 1 bar to avoid look-ahead bias.

Returned format
---------------
MultiIndex DataFrame indexed by (timestamp, symbol) with feature columns.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd


REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


def _to_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to MultiIndex (timestamp, symbol)."""
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        assert set(REQUIRED_COLS).issubset(df.columns), \
            f"Missing required columns: {set(REQUIRED_COLS) - set(df.columns)}"
        return df.sort_index()

    # Flat index + symbol column
    assert "symbol" in df.columns, "Expected a 'symbol' column when index is not MultiIndex."
    assert set(REQUIRED_COLS).issubset(df.columns), \
        f"Missing required columns: {set(REQUIRED_COLS) - set(df.columns)}"
    return (
        df
        .set_index(["symbol"], append=True)   # index -> (timestamp, symbol)
        .swaplevel(0, 1)
        .sort_index()
    )


def _groupby_symbol(mi_df: pd.DataFrame):
    return mi_df.groupby(level=1, sort=False)


def _safe_shifted_roll(series: pd.Series, window: int, fn: str) -> pd.Series:
    """Apply rolling fn then shift by 1 to avoid leakage."""
    rolled = getattr(series.rolling(window, min_periods=max(2, window // 2)), fn)()
    return rolled.shift(1)


def _returns(close: pd.Series, windows: Iterable[int]) -> pd.DataFrame:
    out = {}
    for w in windows:
        out[f"ret_{w}"] = close.pct_change(w).shift(1)
    return pd.DataFrame(out, index=close.index)


def _volatility(close: pd.Series, windows: Iterable[int]) -> pd.DataFrame:
    logret = np.log(close).diff()
    out = {}
    for w in windows:
        out[f"vol_{w}"] = _safe_shifted_roll(logret, w, "std") * np.sqrt(252.0)
    return pd.DataFrame(out, index=close.index)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    roll_up = _safe_shifted_roll(up, window, "mean")
    roll_dn = _safe_shifted_roll(dn, window, "mean")
    rs = roll_up / (roll_dn + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.rename(f"rsi_{window}")


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    # EMA with adjust=False to mimic trading platforms; shift to avoid leakage
    ema_fast = close.ewm(span=fast, adjust=False).mean().shift(1)
    ema_slow = close.ewm(span=slow, adjust=False).mean().shift(1)
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return pd.DataFrame({
        f"macd_{fast}_{slow}": macd,
        f"macd_signal_{signal}": sig,
        f"macd_hist_{fast}_{slow}_{signal}": hist
    })


def _vwap_distance(df: pd.DataFrame, window: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (tp * df["volume"]).rolling(window, min_periods=max(2, window // 2)).sum().shift(1)
    vv = df["volume"].rolling(window, min_periods=max(2, window // 2)).sum().shift(1)
    vwap = pv / (vv + 1e-12)
    return ((df["close"] / vwap) - 1.0).rename(f"dist_vwap_{window}")


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = _safe_shifted_roll(tr, window, "mean")
    return atr.rename(f"atr_{window}")


def _cross_sectional_zscore(mi_df: pd.DataFrame, col: str, by: str = "date") -> pd.Series:
    """
    Cross-sectional z-score of column 'col' computed per timestamp.
    Assumes MultiIndex. 'by' ignored (kept for API symmetry).
    """
    x = mi_df[col]
    # group by timestamp (level=0)
    grp = x.groupby(level=0)
    mu = grp.transform("mean")
    sd = grp.transform("std").replace(0, np.nan)
    return ((x - mu) / sd).rename(col + "_xz")


def build_features(
    ohlcv: pd.DataFrame,
    return_windows: Iterable[int] = (1, 5, 20),
    vol_windows: Iterable[int] = (5, 20, 60),
    rsi_window: int = 14,
    macd_params: Tuple[int, int, int] = (12, 26, 9),
    vwap_window: int = 20,
    atr_window: int = 14,
    include_cross_sectionals: bool = True,
) -> pd.DataFrame:
    """
    Build a standard feature set from OHLCV data.

    Parameters
    ----------
    ohlcv : DataFrame
        Input price data (see module docstring for formats).
    return_windows : iterable
        Lags for simple returns (shifted to prevent leakage).
    vol_windows : iterable
        Windows for realized (log-return) volatility (annualized).
    rsi_window : int
        RSI lookback.
    macd_params : (fast, slow, signal)
        EMA spans for MACD.
    vwap_window : int
        Rolling VWAP window.
    atr_window : int
        ATR lookback.
    include_cross_sectionals : bool
        If True, compute cross-sectional z-scores for selected features.

    Returns
    -------
    DataFrame
        MultiIndex (timestamp, symbol) x features.
    """
    df = _to_multiindex(ohlcv)

    # Build per-symbol features
    feats = []
    g = _groupby_symbol(df)
    for sym, gdf in g:
        # Ensure time-sorted per symbol
        gdf = gdf.sort_index(level=0)

        ret = _returns(gdf["close"], return_windows)
        vol = _volatility(gdf["close"], vol_windows)
        rsi = _rsi(gdf["close"], rsi_window)
        macd = _macd(gdf["close"], *macd_params)
        vwapd = _vwap_distance(gdf, vwap_window)
        atr = _atr(gdf, atr_window)

        sym_feats = pd.concat([ret, vol, rsi, macd, vwapd, atr], axis=1)
        # sym_feats["symbol"] = sym # this makes another symbol when it was already in the df from the index
        feats.append(sym_feats)

    out = pd.concat(feats, axis=0)
    out.index.name = None
    out = out.set_index("symbol", append=True).swaplevel(0, 1).sort_index()

    # Cross-sectional (per timestamp) transforms
    if include_cross_sectionals:
        for col in ["ret_5", "ret_20", "vol_20", "vol_60", f"rsi_{rsi_window}"]:
            if col in out.columns:
                out[col + "_rank"] = out[col].groupby(level=0).rank(pct=True)
                out[col + "_xz"] = _cross_sectional_zscore(out, col)

    # Housekeeping
    out = out.replace([np.inf, -np.inf], np.nan)
    # Keep at least some data present; user can dropna further as needed.
    return out


if __name__ == "__main__":
    # Quick self-test with synthetic data
    idx = pd.date_range("2022-01-01", periods=300, freq="D")
    syms = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(0)
    rows = []
    for s in syms:
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
        high = prices * (1 + rng.normal(0.001, 0.01, len(idx)).clip(-0.01, 0.03))
        low = prices * (1 - rng.normal(0.001, 0.01, len(idx)).clip(-0.03, 0.01))
        vol = rng.integers(1e5, 3e5, len(idx))
        rows.append(pd.DataFrame(
            {"open": prices, "high": high, "low": low, "close": prices, "volume": vol},
            index=pd.MultiIndex.from_product([[s], idx], names=["symbol", "timestamp"])
        ))
    ohlcv_demo = pd.concat(rows).swaplevel(0, 1).sort_index()
    feats_demo = build_features(ohlcv_demo)
    print(feats_demo.head())
