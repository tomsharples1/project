import os
import pandas as pd
import yfinance as yf

from features.build_features import build_features
from features.label_triple_barrier import triple_barrier_labels, forward_return_labels

# Paths
os.makedirs("data", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# ---------- A) Download data ----------
tickers = ["AAPL", "MSFT", "GOOG"]
dfy = yf.download(tickers, start="2020-01-01", end="2024-12-31", interval="1d", group_by="ticker", auto_adjust=False)

# Reshape to MultiIndex (timestamp, symbol) with lowercase OHLCV
parts = []
for sym in tickers:
    tmp = dfy[sym].copy()
    tmp.columns = tmp.columns.str.lower()  # open, high, low, close, adj close, volume
    tmp["symbol"] = sym
    tmp = tmp.reset_index().set_index(["Date", "symbol"]).rename_axis(["timestamp", "symbol"])
    parts.append(tmp)
ohlcv = pd.concat(parts).sort_index()

# ---------- B) Build features ----------
X = build_features(ohlcv)
X.to_parquet("data/features.parquet", engine='fastparquet', index=True)

# ---------- C) Labels ----------
# Option 1: classification via triple barrier (drop timeouts later)
labels_cls = triple_barrier_labels(ohlcv, up_mult=1.0, down_mult=1.0, vol_lookback=50, max_holding=10)
labels_cls.to_parquet("data/labels_cls.parquet", engine='fastparquet', index=True)

# Option 2: regression via forward returns (e.g., 5-day ahead)
y_reg = forward_return_labels(ohlcv, horizon=5)
y_reg.to_frame("y").to_parquet("data/labels_reg.parquet", engine='fastparquet', index=True)

print("Saved:")
print("  data/features.parquet")
print("  data/labels_cls.parquet (columns: t1, label, ret)")
print("  data/labels_reg.parquet (column: y)")
