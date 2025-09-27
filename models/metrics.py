# models/metrics.py
"""
Standalone evaluation helpers for predictions vs realized outcomes.

- Classification: ROC-AUC, PR-AUC, F1, log-loss, calibration bins.
- Regression: RMSE/MAE/R2.
- Rank-decile analysis (common in cross-sectional alphas).

Example
-------
python -m models.metrics \
  --y_true_path data/y_true.parquet \
  --y_pred_path artifacts/preds.parquet \
  --mode classification \
  --positive_class 1 \
  --report_out artifacts/metrics.json
"""

from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, log_loss,
    mean_squared_error, mean_absolute_error, r2_score
)

def read_df(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, index_col=0)
    raise ValueError(f"Unsupported file type: {path}")

def align_on_index(a: pd.Series | pd.DataFrame, b: pd.Series | pd.DataFrame):
    return a.join(b, how="inner")

def classification_report(y_true: pd.Series, pred: pd.DataFrame | pd.Series, positive_class: int | None = None) -> dict:
    out = {}

    if isinstance(pred, pd.DataFrame) and pred.shape[1] > 1:
        # multiclass
        proba = pred.values
        classes = [int(c.replace("proba_", "")) for c in pred.columns]
        y = y_true.values
        out["log_loss"] = float(log_loss(y, proba, labels=classes))
        # derive macro F1 from argmax
        yhat = proba.argmax(1)
        out["f1_macro"] = float(f1_score(y, yhat, average="macro"))
    else:
        # binary: pred is a Series with proba of positive class
        p = pred.squeeze().values
        y = y_true.values
        if positive_class is None:
            positive_class = int(np.nanmax(np.unique(y)))
        y_bin = (y == positive_class).astype(int)
        out["auc_roc"] = float(roc_auc_score(y_bin, p))
        out["auc_pr"] = float(average_precision_score(y_bin, p))
        out["f1_at_0.5"] = float(f1_score(y_bin, (p >= 0.5).astype(int)))
        out["log_loss"] = float(log_loss(y_bin, np.vstack([1 - p, p]).T))
    return out

def regression_report(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

def decile_analysis(scores: pd.Series, fwd_returns: pd.Series, k: int = 10) -> pd.DataFrame:
    """
    Cross-sectional decile buckets by timestamp; compute average forward return per bucket.
    """
    df = pd.concat({"score": scores, "fwd": fwd_returns}, axis=1).dropna()
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Decile analysis expects MultiIndex (timestamp, symbol).")
    def per_ts(g):
        q = pd.qcut(g["score"].rank(method="first"), k, labels=False)  # 0..k-1
        return pd.DataFrame({"decile": q, "fwd": g["fwd"]})
    binned = df.groupby(level=0).apply(per_ts).reset_index(level=0, drop=True)
    return binned.groupby("decile")["fwd"].mean().to_frame("avg_fwd_return")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--y_true_path", required=True, help="Parquet/CSV with ground truth values (index-aligned).")
    p.add_argument("--y_pred_path", required=True, help="Parquet/CSV with predictions from infer.py.")
    p.add_argument("--mode", choices=["classification", "regression"], required=True)
    p.add_argument("--positive_class", type=int, default=None)
    p.add_argument("--report_out", required=True)
    p.add_argument("--decile", action="store_true", help="If set, compute decile analysis (requires fwd returns).")
    p.add_argument("--fwd_ret_col", default="y", help="Column name for forward returns if doing decile analysis.")
    return p.parse_args()

def main():
    args = parse_args()
    y_true_df = read_df(args.y_true_path)
    y_pred_df = read_df(args.y_pred_path)

    # Normalize to Series/DataFrame shapes we expect
    if isinstance(y_true_df, pd.DataFrame) and "y" in y_true_df.columns:
        y_true = y_true_df["y"]
    elif isinstance(y_true_df, pd.Series):
        y_true = y_true_df
    elif isinstance(y_true_df, pd.DataFrame) and "label" in y_true_df.columns:
        y_true = y_true_df["label"]
    else:
        # take first column
        y_true = y_true_df.iloc[:, 0]

    # Align on index
    data = align_on_index(y_true.to_frame("y_true"), y_pred_df)
    y_true = data["y_true"]

    if args.mode == "classification":
        # Identify prediction shape
        if isinstance(y_pred_df, pd.DataFrame) and y_pred_df.shape[1] > 1:
            # multiclass
            pred = data[y_pred_df.columns]
        else:
            # binary
            col = "proba_pos" if "proba_pos" in data.columns else data.columns[0]
            pred = data[col]
        report = classification_report(y_true, pred, args.positive_class)
    else:
        col = "y_pred" if "y_pred" in data.columns else data.columns[0]
        report = regression_report(y_true, data[col])

    # Optional decile analysis (for cross-sectional alphas)
    if args.decile:
        if args.mode == "classification":
            # use score = proba_pos if available, else max class prob
            if isinstance(y_pred_df, pd.DataFrame) and "proba_pos" in y_pred_df.columns:
                score = data["proba_pos"]
            elif isinstance(y_pred_df, pd.DataFrame):
                score = data[y_pred_df.columns].max(axis=1)
            else:
                score = data.iloc[:, 1] if data.shape[1] > 1 else data.iloc[:, 0]
        else:
            score = data[col]
        # Need forward returns; try to get from y_true_df
        if isinstance(y_true_df, pd.DataFrame) and args.fwd_ret_col in y_true_df.columns:
            fwd = y_true_df[args.fwd_ret_col]
        else:
            fwd = y_true  # fallback
        # align again
        both = align_on_index(score.to_frame("score"), fwd.to_frame(args.fwd_ret_col))
        deciles = decile_analysis(both["score"], both[args.fwd_ret_col])
        report["decile_avg_fwd"] = deciles["avg_fwd_return"].to_dict()

    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
