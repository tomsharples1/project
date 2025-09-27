# models/train.py
"""
Train LightGBM models for time-series tabular alpha.

- Handles binary/multiclass classification (e.g., triple-barrier) and regression.
- Purged, embargoed rolling CV to avoid leakage.
- Saves model + feature metadata for safe inference.

Example
-------
python -m models.train \
  --features_path data/features.parquet \
  --labels_path data/labels.parquet \
  --task classification \
  --label_col label \
  --drop_class_zero \
  --n_splits 4 \
  --embargo 5 \
  --model_out artifacts/model_lgbm.pkl \
  --meta_out artifacts/model_meta.json
"""

from __future__ import annotations
import argparse, json, os, sys
from dataclasses import dataclass, asdict
from typing import Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, log_loss,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier, LGBMRegressor

# ---------- Utilities ----------
def read_df(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")

def align_xy(X: pd.DataFrame, y: pd.DataFrame | pd.Series, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Inner-join on index; return aligned (X, y)."""
    if isinstance(y, pd.DataFrame):
        if label_col not in y.columns:
            raise ValueError(f"Label column '{label_col}' not in labels DataFrame.")
        y = y[label_col]
    both = X.join(y.to_frame("y"), how="inner")
    cols = [c for c in both.columns if c != "y"]
    X_aligned = both[cols]
    y_aligned = both["y"]
    return X_aligned, y_aligned

def infer_time_index(X: pd.DataFrame) -> pd.Series:
    """
    Return a DatetimeIndex aligned with rows for CV splitting.
    Works with MultiIndex in any order; prefers a level literally named
    'timestamp' or the level that most successfully parses as datetime.
    """
    idx = X.index
    if isinstance(idx, pd.MultiIndex):
        # 1) Prefer level named 'timestamp'
        if idx.names and "timestamp" in idx.names:
            lvl = idx.names.index("timestamp")
            ts = pd.to_datetime(idx.get_level_values(lvl), errors="coerce")
            return ts
        # 2) Otherwise, pick the level with the most valid datetimes
        best_ts, best_count = None, -1
        for lvl in range(idx.nlevels):
            cand = pd.to_datetime(idx.get_level_values(lvl), errors="coerce")
            count = cand.notna().sum()
            if count > best_count:
                best_ts, best_count = cand, count
        if best_ts is None or best_ts.notna().sum() == 0:
            raise ValueError("Could not infer a datetime-like index level.")
        return best_ts
    elif isinstance(idx, pd.DatetimeIndex):
        return idx
    else:
        return pd.to_datetime(idx, errors="raise")


def unique_sorted_times(ts: pd.Series) -> np.ndarray:
    return np.array(pd.Index(ts).unique().sort_values())

def time_series_purged_splits(
    times: pd.Series,
    n_splits: int = 4,
    embargo: int = 5
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Rolling expanding window CV:
      - Split by unique timestamps (NOT by row count).
      - Purge an 'embargo' window between train and validation.
    Yields (train_idx, val_idx) arrays of row indices.
    """
    uniq = unique_sorted_times(times)
    if n_splits < 1 or len(uniq) < (n_splits + 1):
        raise ValueError("Not enough unique timestamps for the requested n_splits.")

    fold = len(uniq) // (n_splits + 1)
    for i in range(n_splits):
        train_end_t = uniq[(i + 1) * fold - 1]
        val_start_pos = (i + 1) * fold + embargo
        val_end_pos = min(val_start_pos + fold, len(uniq))
        if val_start_pos >= val_end_pos:
            break
        val_start_t, val_end_t = uniq[val_start_pos], uniq[val_end_pos - 1]

        train_mask = times <= train_end_t
        val_mask = (times >= val_start_t) & (times <= val_end_t)
        yield np.where(train_mask)[0], np.where(val_mask)[0]


# ---------- Metrics (quick) ----------

def metrics_classification(y_true: np.ndarray, proba: np.ndarray, labels: List[int]) -> dict:
    """
    For binary: proba is P(class1). For multiclass: proba is (n_samples, n_classes).
    """
    out = {}
    if len(labels) == 2:
        # treat max(labels) as positive by default
        pos_label = max(labels)
        # Build hard preds at 0.5
        y_pred = (proba >= 0.5).astype(int) if proba.ndim == 1 else proba.argmax(1)
        y_bin = (np.array(y_true) == pos_label).astype(int)
        p = proba if proba.ndim == 1 else proba[:, labels.index(pos_label)]

        out["auc_roc"] = float(roc_auc_score(y_bin, p))
        out["auc_pr"] = float(average_precision_score(y_bin, p))
        out["f1"] = float(f1_score(y_bin, (p >= 0.5).astype(int)))
        out["log_loss"] = float(log_loss(y_bin, np.vstack([1 - p, p]).T))
    else:
        y_pred = proba.argmax(1)
        out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
        out["log_loss"] = float(log_loss(y_true, proba, labels=labels))
    return out

def metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ---------- Training ----------

@dataclass
class TrainConfig:
    task: str  # "classification" or "regression"
    label_col: str
    drop_class_zero: bool
    n_splits: int
    embargo: int
    learning_rate: float
    n_estimators: int
    # num_leaves: int
    subsample: float
    # colsample_bytree: float
    # reg_alpha: float
    # reg_lambda: float
    random_state: int
    force_col_wise: bool

def train(X: pd.DataFrame, y: pd.Series, cfg: TrainConfig) -> dict:
    # Optional: drop "0" class (timeouts) for triple-barrier
    if cfg.task == "classification" and cfg.drop_class_zero:
        mask = y != 0
        X, y = X.loc[mask], y.loc[mask]

    mask_y = y.notna()
    X, y = X.loc[mask_y], y.loc[mask_y]

    min_non_null = int(0.2 * X.shape[1])  # at least 20% of features not NaN
    row_ok = X.notna().sum(axis=1) >= min_non_null
    X, y = X.loc[row_ok], y.loc[row_ok]

    times = infer_time_index(X)
    splits = list(time_series_purged_splits(times, cfg.n_splits, cfg.embargo))

    if cfg.task == "classification":
        classes = sorted(np.unique(y.dropna()))
        objective = "binary" if len(classes) == 2 else "multiclass"
        base_model = LGBMClassifier(
            objective=objective,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            # num_leaves=cfg.num_leaves,
            subsample=cfg.subsample,
            # colsample_bytree=cfg.colsample_bytree,
            # reg_alpha=cfg.reg_alpha,
            # reg_lambda=cfg.reg_lambda,
            random_state=cfg.random_state,
            force_col_wise = cfg.force_col_wise, 
            n_jobs=-1
        )
    else:
        base_model = LGBMRegressor(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            # num_leaves=cfg.num_leaves,
            subsample=cfg.subsample,
            # colsample_bytree=cfg.colsample_bytree,
            # reg_alpha=cfg.reg_alpha,
            # reg_lambda=cfg.reg_lambda,
            random_state=cfg.random_state,
            force_col_wise = cfg.force_col_wise,
            n_jobs=-1
        )

    model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", base_model)
    ])  

    # Cross-validated training (rolling)
    cv_scores = []
    for tr, va in splits:
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        Xva, yva = X.iloc[va], y.iloc[va]
        model.fit(Xtr, ytr)
        if cfg.task == "classification":
            proba = model.predict_proba(Xva)
            if proba.ndim == 2 and proba.shape[1] == 2:
                proba_vec = proba[:, 1]
                s = metrics_classification(yva.values, proba_vec, labels=sorted(np.unique(y)))
            else:
                s = metrics_classification(yva.values, proba, labels=sorted(np.unique(y)))
        else:
            pred = model.predict(Xva)
            s = metrics_regression(yva.values, pred)
        cv_scores.append(s)

    # Fit on all data at the end
    model.fit(X, y)

    # Aggregate CV metrics
    agg = {}
    for k in cv_scores[0].keys():
        agg[k] = float(np.mean([d[k] for d in cv_scores]))

    return {
        "model": model,
        "cv_metrics": agg,
        "classes": [int(c) for c in sorted(np.unique(y))] if cfg.task == "classification" else None, 
        "feature_names": list(X.columns)
    }


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_path", required=True)
    p.add_argument("--labels_path", required=True)
    p.add_argument("--task", choices=["classification", "regression"], required=True)
    p.add_argument("--label_col", default="label")
    p.add_argument("--drop_class_zero", action="store_true")
    p.add_argument("--n_splits", type=int, default=4)
    p.add_argument("--embargo", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=0.03)
    p.add_argument("--n_estimators", type=int, default=1000)
    p.add_argument("--num_leaves", type=int, default=63)
    p.add_argument("--subsample", type=float, default=0.7)
    p.add_argument("--colsample_bytree", type=float, default=0.7)
    p.add_argument("--reg_alpha", type=float, default=1.0)
    p.add_argument("--reg_lambda", type=float, default=1.0)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--model_out", required=True)
    p.add_argument("--meta_out", required=True)
    return p.parse_args()

@dataclass
class ModelMeta:
    task: str
    classes: List[int] | None
    feature_names: List[str]
    label_col: str
    drop_class_zero: bool
    cv_metrics: dict

def main():
    args = parse_args()
    X = read_df(args.features_path)
    y = read_df(args.labels_path)

    X, y_vec = align_xy(X, y, args.label_col