# models/infer.py
"""
Batch inference on feature frames with a saved model + metadata.

Example
-------
python -m models.infer \
  --features_path data/features_live.parquet \
  --model_path artifacts/model_lgbm.pkl \
  --meta_path artifacts/model_meta.json \
  --pred_out artifacts/preds.parquet
"""

from __future__ import annotations
import argparse, json
import joblib
import numpy as np
import pandas as pd


def read_df(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--meta_path", required=True)
    p.add_argument("--pred_out", required=True)
    return p.parse_args()

def main():
    args = parse_args()
    X = read_df(args.features_path)
    meta = json.load(open(args.meta_path, "r"))
    model = joblib.load(args.model_path)

    # Align feature columns (order and missing)
    missing = [c for c in meta["feature_names"] if c not in X.columns]
    if missing:
        # Create missing as NaN
        for m in missing:
            X[m] = np.nan
    # Extra columns are ignored
    X = X[meta["feature_names"]]
    X = X.replace([np.inf, -np.inf], np.nan)

    if meta["task"] == "classification":
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            preds = pd.Series(proba[:, 1], index=X.index, name="proba_pos")
        else:
            # multiclass: output per-class probabilities
            preds = pd.DataFrame(proba, index=X.index, columns=[f"proba_{c}" for c in meta["classes"]])
    else:
        yhat = model.predict(X)
        preds = pd.Series(yhat, index=X.index, name="y_pred")

    # Save
    if args.pred_out.lower().endswith(".parquet"):
        if isinstance(preds, pd.Series):
            preds.to_frame().to_parquet(args.pred_out)
        else:
            preds.to_parquet(args.pred_out)
    else:
        if isinstance(preds, pd.Series):
            preds.to_frame().to_csv(args.pred_out, index=True)
        else:
            preds.to_csv(args.pred_out, index=True)

    print(f"Saved predictions -> {args.pred_out}")

if __name__ == "__main__":
    main()
