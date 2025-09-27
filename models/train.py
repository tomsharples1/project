# models/train.py
from __future__ import annotations
import argparse, json, os
from dataclasses import dataclass, asdict
from typing import Iterable, List, Tuple, Optional

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
        return pd.read_csv(path, index_col=0 if path.lower().endswith(".csv") else None)
    raise ValueError(f"Unsupported file type: {path}")

def align_xy(X: pd.DataFrame, y: pd.DataFrame | pd.Series, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if isinstance(y, pd.DataFrame):
        if label_col not in y.columns:
            raise ValueError(f"Label column '{label_col}' not in labels DataFrame.")
        y = y[label_col]
    both = X.join(y.to_frame("y"), how="inner")
    cols = [c for c in both.columns if c != "y"]
    return both[cols], both["y"]

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

def unique_sorted_times(ts: pd.Series) -> np.ndarray:
    # FORCE datetime64[ns], drop NaT, return a proper DatetimeIndex-backed ndarray
    ts = pd.to_datetime(pd.Series(ts), errors="coerce")
    ts = ts[ts.notna()]
    return pd.DatetimeIndex(ts).unique().sort_values().to_numpy(dtype="datetime64[ns]")

def time_series_purged_splits(times: pd.Series, n_splits: int = 4, embargo: int = 5) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
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

def apply_time_holdout(
    times: pd.Series,
    test_split_date: Optional[str],
    test_frac: Optional[float],
    embargo: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build boolean masks for train/test by time, with an optional embargo.
    - If test_split_date is given: test = ts >= that date
    - Else if test_frac is given: test = last fraction of unique timestamps
    - Embargo removes the last `embargo` unique timestamps from the train set
      immediately before the test start.
    """
    # 1) coerce to datetime and keep alignment
    ts = pd.to_datetime(times, errors="coerce")
    if not ts.index.equals(times.index):
        ts.index = times.index

    # drop rows with invalid time (rare)
    valid = ts.notna()
    ts = ts[valid]

    # 2) decide test start
    uniq = pd.DatetimeIndex(ts).unique().sort_values()
    if len(uniq) == 0:
        raise ValueError("No valid timestamps to split on.")

    if test_split_date:
        test_start = pd.Timestamp(test_split_date)
    else:
        if not (test_frac and 0 < float(test_frac) < 1):
            raise ValueError("Provide --test_split_date or a 0<--test_frac<1.")
        cut_idx = int(len(uniq) * (1.0 - float(test_frac)))
        cut_idx = min(max(cut_idx, 0), len(uniq) - 1)
        test_start = uniq[cut_idx]

    # 3) embargo handling
    if embargo > 0:
        # position of test_start in uniq
        pos = uniq.searchsorted(test_start)
        embargo_pos = max(0, pos - embargo)
        train_end_time = uniq[embargo_pos - 1] if embargo_pos > 0 else pd.Timestamp.min
        train_sel = ts <= train_end_time
    else:
        train_sel = ts < test_start

    test_sel = ts >= test_start

    # 4) map back to full index (rows with NaT become False in both masks)
    train_mask = pd.Series(False, index=times.index)
    test_mask  = pd.Series(False, index=times.index)
    train_mask.loc[train_sel.index] = train_sel.values
    test_mask.loc[test_sel.index]   = test_sel.values

    return train_mask.values, test_mask.values

# ---------- Metrics ----------
def metrics_classification(y_true: np.ndarray, proba: np.ndarray, labels: List[int]) -> dict:
    out = {}
    if len(labels) == 2:
        pos_label = max(labels)
        if proba.ndim == 1:
            p = proba
        else:
            p = proba[:, labels.index(pos_label)]
        y_bin = (np.array(y_true) == pos_label).astype(int)
        out["auc_roc"] = float(roc_auc_score(y_bin, p))
        out["auc_pr"] = float(average_precision_score(y_bin, p))
        out["f1_at_0.5"] = float(f1_score(y_bin, (p >= 0.5).astype(int)))
        out["log_loss"] = float(log_loss(y_bin, np.vstack([1 - p, p]).T))
    else:
        y_pred = proba.argmax(1)
        out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
        out["log_loss"] = float(log_loss(y_true, proba, labels=labels))
    return out

def metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # Backward-compatible RMSE
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(rmse),
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
    subsample: float
    random_state: int
    force_col_wise: bool
    test_split_date: Optional[str]
    test_frac: Optional[float]
    save_test_prefix: Optional[str]
    eval_test_now: bool

def train(X: pd.DataFrame, y: pd.Series, cfg: TrainConfig) -> dict:
    
    if cfg.task == "classification" and cfg.drop_class_zero:
        ok = y != 0
        X, y = X.loc[ok], y.loc[ok]

    X = X.replace([np.inf, -np.inf], np.nan)
    y = y[y.notna()]
    X = X.loc[y.index]                 # align

    # require at least 20% non-NaN features
    min_non_null = max(1, int(0.2 * X.shape[1]))
    row_ok = X.notna().sum(axis=1) >= min_non_null
    X, y = X.loc[row_ok], y.loc[row_ok]

    # Build time series
    times_all = infer_time_index(X)

    # Split
    tr_mask, te_mask = apply_time_holdout(times_all, cfg.test_split_date, cfg.test_frac, cfg.embargo)
    Xtr, ytr = X.loc[tr_mask], y.loc[tr_mask]
    Xte, yte = X.loc[te_mask], y.loc[te_mask]


    # CV only on the train period
    splits = list(time_series_purged_splits(infer_time_index(Xtr), cfg.n_splits, cfg.embargo))

    # Build model (with imputer)
    if cfg.task == "classification":
        classes = sorted(np.unique(ytr.dropna()))
        objective = "binary" if len(classes) == 2 else "multiclass"
        base = LGBMClassifier(
            objective=objective,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            random_state=cfg.random_state,
            force_col_wise=cfg.force_col_wise,
            n_jobs=-1,
        )
    else:
        base = LGBMRegressor(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            random_state=cfg.random_state,
            force_col_wise=cfg.force_col_wise,
            n_jobs=-1,
        )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", base),
    ])

    # ---- Rolling CV on train subset
    cv_scores = []
    for tr, va in splits:
        X_a, y_a = Xtr.iloc[tr], ytr.iloc[tr]
        X_b, y_b = Xtr.iloc[va], ytr.iloc[va]
        model.fit(X_a, y_a)
        if cfg.task == "classification":
            proba = model.predict_proba(X_b)
            if proba.ndim == 2 and proba.shape[1] == 2:
                s = metrics_classification(y_b.values, proba[:, 1], labels=sorted(np.unique(ytr)))
            else:
                s = metrics_classification(y_b.values, proba, labels=sorted(np.unique(ytr)))
        else:
            pred = model.predict(X_b)
            # mask NaNs just in case
            m = np.isfinite(y_b.values) & np.isfinite(pred)
            if m.sum() == 0:
                continue
            s = metrics_regression(y_b.values[m], pred[m])
        cv_scores.append(s)

    # Fit on all train data
    model.fit(Xtr, ytr)

    # Aggregate CV metrics
    cv_agg = {k: float(np.mean([d[k] for d in cv_scores])) for k in cv_scores[0].keys()} if cv_scores else {}

    # Optional test evaluation
    test_metrics = None
    if cfg.eval_test_now and len(Xte) > 0:
        if cfg.task == "classification":
            proba = model.predict_proba(Xte)
            if proba.ndim == 2 and proba.shape[1] == 2:
                test_metrics = metrics_classification(yte.values, proba[:, 1], labels=sorted(np.unique(ytr)))
            else:
                test_metrics = metrics_classification(yte.values, proba, labels=sorted(np.unique(ytr)))
        else:
            pred = model.predict(Xte)
            m = np.isfinite(yte.values) & np.isfinite(pred)
            test_metrics = metrics_regression(yte.values[m], pred[m])

    # Save optional test sets
    if cfg.save_test_prefix:
        # Keep original indices so you can reuse later
        Xte.to_parquet(f"{cfg.save_test_prefix}_features.parquet")
        yte.to_frame("y").to_parquet(f"{cfg.save_test_prefix}_labels.parquet")

    return {
        "model": model,
        "cv_metrics": cv_agg,
        "test_metrics": test_metrics,
        "classes": ([int(c) for c in sorted(np.unique(ytr))] if cfg.task == "classification" else None),
        "feature_names": list(X.columns),
    }


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_path", required=True)
    p.add_argument("--labels_path", required=True)
    p.add_argument("--task", choices=["classification", "regression"], required=True)
    p.add_argument("--label_col", default="label")
    p.add_argument("--drop_class_zero", action="store_true")

    # CV + leakage control
    p.add_argument("--n_splits", type=int, default=4)
    p.add_argument("--embargo", type=int, default=10)

    # Train/test split
    p.add_argument("--test_split_date", type=str, default=None, help="ISO date. Test starts on/after this date.")
    p.add_argument("--test_frac", type=float, default=None, help="Alternative: last fraction of timestamps as test.")
    p.add_argument("--save_test_prefix", type=str, default=None, help="If set, save test X/y to <prefix>_features.parquet and <prefix>_labels.parquet")
    p.add_argument("--eval_test_now", action="store_true", help="If set, evaluate metrics on test holdout immediately.")

    # Model params (kept simple)
    p.add_argument("--learning_rate", type=float, default=0.03)
    p.add_argument("--n_estimators", type=int, default=1000)
    p.add_argument("--subsample", type=float, default=0.7)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--force_col_wise", action="store_true")

    # Outputs
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
    test_metrics: dict | None
    test_split_date: str | None
    test_frac: float | None
    embargo: int

def main():
    args = parse_args()
    X = read_df(args.features_path)
    y = read_df(args.labels_path)
    X, y_vec = align_xy(X, y, args.label_col)

    cfg = TrainConfig(
        task=args.task,
        label_col=args.label_col,
        drop_class_zero=args.drop_class_zero,
        n_splits=args.n_splits,
        embargo=args.embargo,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        random_state=args.random_state,
        force_col_wise=args.force_col_wise,
        test_split_date=args.test_split_date,
        test_frac=args.test_frac,
        save_test_prefix=args.save_test_prefix,
        eval_test_now=args.eval_test_now,
    )

    result = train(X, y_vec, cfg)

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(result["model"], args.model_out)

    meta = ModelMeta(
        task=args.task,
        classes=result["classes"],
        feature_names=result["feature_names"],
        label_col=args.label_col,
        drop_class_zero=args.drop_class_zero,
        cv_metrics=result["cv_metrics"],
        test_metrics=result["test_metrics"],
        test_split_date=args.test_split_date,
        test_frac=args.test_frac,
        embargo=args.embargo,
    )
    with open(args.meta_out, "w") as f:
        json.dump(asdict(meta), f, indent=2)

    # Pretty print
    out = {"cv_metrics": result["cv_metrics"]}
    if result["test_metrics"] is not None:
        out["test_metrics"] = result["test_metrics"]
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
