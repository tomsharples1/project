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

def infer_time_index(X: pd.DataFrame) -> pd.Series:
    """
    Return a datetime Series aligned to X.index.
    Works for:
      - MultiIndex with ('timestamp', 'symbol') in any order
      - Plain DatetimeIndex
      - Plain index of strings to be parsed as dates
    """
    idx = X.index
    if isinstance(idx, pd.MultiIndex):
        # Prefer a level literally named 'timestamp'
        if idx.names and "timestamp" in idx.names:
            lvl = idx.names.index("timestamp")
            ts = pd.to_datetime(idx.get_level_values(lvl), errors="coerce")
        else:
            # Try each level, pick the one with most valid datetimes
            best_ts, best_ok = None, -1
            for lvl in range(idx.nlevels):
                cand = pd.to_datetime(idx.get_level_values(lvl), errors="coerce")
                ok = cand.notna().sum()
                if ok > best_ok:
                    best_ts, best_ok = cand, ok
            ts = best_ts
    elif isinstance(idx, pd.DatetimeIndex):
        ts = pd.Series(idx, index=idx)
    else:
        ts = pd.to_datetime(pd.Series(idx, index=idx), errors="coerce")

    # Ensure it's a Series indexed exactly like X
    if not isinstance(ts, pd.Series):
        ts = pd.Series(ts, index=X.index)
    else:
        ts.index = X.index
    return ts

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
    uniq = pd.Index(ts).unique().sort_values()
    start_date = meta.get("test_split_date", None)
    test_frac = meta.get("test_frac", None)

    if start_date:
        mask = ts >= pd.Timestamp(start_date)
        return df.loc[mask]

    if test_frac is not None:
        if not (0.0 < float(test_frac) < 1.0):
            raise ValueError(f"Invalid test_frac in meta: {test_frac}")
        cut_idx = int(len(uniq) * (1.0 - float(test_frac)))
        test_start = uniq[cut_idx] if cut_idx < len(uniq) else uniq[-1]
        mask = ts >= pd.Timestamp(test_start)
        return df.loc[mask]

    # If neither present, return unchanged
    return df

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_path", required=True, help="Parquet/CSV of features (can be full set or test-only).")
    p.add_argument("--model_path", required=True)
    p.add_argument("--meta_path", required=True)
    p.add_argument("--pred_out", required=True)

    # Choose test subset
    p.add_argument("--use_test_from_meta", action="store_true",
                   help="Filter features to the test window using test_split_date/test_frac from meta.json.")
    p.add_argument("--start_date", type=str, default=None, help="Override: filter rows with timestamp >= start_date.")
    p.add_argument("--end_date", type=str, default=None, help="Override: filter rows with timestamp <= end_date.")
    return p.parse_args()

def main():
    args = parse_args()

    # Load
    X = apply_meta_test_filter(X, meta)

    with open(args.meta_path, "r") as fh:
        meta = json.load(fh)
    model = joblib.load(args.model_path)
    
    # Explicit date bounds have priority if provided
    if args.start_date or args.end_date:
        X = filter_by_time(X, args.start_date, args.end_date)

    # Align feature columns (order & missing)
    feat_names = meta["feature_names"]
    missing = [c for c in feat_names if c not in X.columns]
    for m in missing:
        X[m] = np.nan
    X = X[feat_names]
    X = X.replace([np.inf, -np.inf], np.nan)

    # Predict using the stored task type
    task = meta["task"]
    if task == "classification":
        proba = model.predict_proba(X)
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] == 2:
            preds = pd.Series(proba[:, 1], index=X.index, name="proba_pos")
        else:
            classes = meta.get("classes", [])
            preds = pd.DataFrame(proba, index=X.index, columns=[f"proba_{c}" for c in classes])
    else:
        yhat = model.predict(X)
        preds = pd.Series(yhat, index=X.index, name="y_pred")

    # Save
    if args.pred_out.lower().endswith(".parquet"):
        (preds.to_frame() if isinstance(preds, pd.Series) else preds).to_parquet(args.pred_out)
    else:
        (preds.to_frame() if isinstance(preds, pd.Series) else preds).to_csv(args.pred_out, index=True)

    print(f"Saved predictions -> {args.pred_out}")
    if args.use_test_from_meta or args.start_date or args.end_date:
        # Small sanity echo
        ts = infer_time_index(X)
        print(f"Rows predicted: {len(X)} | time range: {ts.min()} .. {ts.max()}")

if __name__ == "__main__":
    main()
