# models/infer.py
from __future__ import annotations
import argparse, json
import joblib
import numpy as np
import pandas as pd


def read_df(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    if path.lower().endswith(".csv"):
        # keep index if present
        try:
            return pd.read_csv(path, index_col=0)
        except Exception:
            return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")

def infer_time_index(df: pd.DataFrame) -> pd.Series:
    """Return a DatetimeIndex aligned with rows, tolerating (timestamp, symbol) or (symbol, timestamp)."""
    idx = df.index
    if isinstance(idx, pd.MultiIndex):
        # Prefer a level literally called 'timestamp'
        if idx.names and "timestamp" in idx.names:
            lvl = idx.names.index("timestamp")
            return pd.to_datetime(idx.get_level_values(lvl), errors="coerce")
        # Otherwise pick the most datetime-like level
        best_ts, best_count = None, -1
        for lvl in range(idx.nlevels):
            cand = pd.to_datetime(idx.get_level_values(lvl), errors="coerce")
            cnt = cand.notna().sum()
            if cnt > best_count:
                best_ts, best_count = cand, cnt
        if best_ts is None or best_ts.notna().sum() == 0:
            raise ValueError("Could not infer a datetime-like index level for features.")
        return best_ts
    if isinstance(idx, pd.DatetimeIndex):
        return idx
    return pd.to_datetime(idx, errors="raise")

def filter_by_time(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """Return rows with timestamp in [start, end] (inclusive), if provided."""
    ts = infer_time_index(df)
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= ts >= pd.Timestamp(start)
    if end is not None:
        mask &= ts <= pd.Timestamp(end)
    return df.loc[mask]

def apply_meta_test_filter(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Use meta.json fields to select the held-out test slice:
      - If test_split_date present: timestamp >= test_split_date
      - Else if test_frac present: last fraction of unique timestamps
      - Note: we do not re-apply embargo here; train already embargoed the split boundary.
    """
    ts = infer_time_index(df)
    uniq = pd.Index(ts).unique().sort_values