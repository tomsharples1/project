# models/advanced_train.py
"""
Advanced model training for algorithmic trading with multiple prediction targets.
Supports ensemble methods, multi-task learning, and sophisticated validation.
"""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Multiple model types
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, Ridge

# For ensemble methods
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.model_selection import cross_val_score


@dataclass
class AdvancedTrainConfig:
    """Advanced training configuration."""
    # Data settings
    features_path: str
    labels_path: str
    prediction_target: str = 'ranking'  # ranking, regime, reversion, earnings
    label_column: str = 'rank_10d'
    
    # Model settings
    model_types: List[str] = None  # ['lightgbm', 'xgboost', 'random_forest']
    use_ensemble: bool = True
    ensemble_method: str = 'voting'  # voting, stacking, blending
    
    # Training settings
    test_frac: float = 0.2
    val_frac: float = 0.1
    n_cv_folds: int = 5
    embargo_days: int = 5
    
    # Feature engineering
    feature_selection: bool = True
    max_features: Optional[int] = 100
    remove_correlated: bool = True
    correlation_threshold: float = 0.95
    
    # Output paths
    model_output_dir: str = 'models'
    experiment_name: str = 'advanced_trading_model'

    def __post_init__(self):
        if self.model_types is None:
            self.model_types = ['lightgbm', 'xgboost', 'random_forest']


def create_time_series_splits(
    timestamps: pd.Series, 
    n_splits: int = 5, 
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    embargo_days: int = 5
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create train/val/test splits for time series data with embargo.
    Returns list of (train_idx, val_idx, test_idx) tuples.
    """
    unique_dates = pd.to_datetime(timestamps).unique()
    unique_dates = np.sort(unique_dates)
    
    n_dates = len(unique_dates)
    test_size = int(n_dates * test_frac)
    val_size = int(n_dates * val_frac)
    
    splits = []
    
    for i in range(n_splits):
        # Calculate split points
        split_point = int(n_dates * (0.5 + 0.1 * i))  # Progressive splits
        val_start = min(split_point, n_dates - test_size - val_size - embargo_days)
        val_end = val_start + val_size
        test_start = val_end + embargo_days
        test_end = min(test_start + test_size, n_dates)
        
        if val_start <= 0 or test_start >= n_dates:
            continue
            
        # Get date ranges
        train_end_date = unique_dates[val_start - 1]
        val_start_date = unique_dates[val_start]
        val_end_date = unique_dates[val_end - 1]
        test_start_date = unique_dates[test_start]
        test_end_date = unique_dates[test_end - 1]
        
        # Convert to indices
        train_mask = pd.to_datetime(timestamps) <= train_end_date
        val_mask = (pd.to_datetime(timestamps) >= val_start_date) & (pd.to_datetime(timestamps) <= val_end_date)
        test_mask = (pd.to_datetime(timestamps) >= test_start_date) & (pd.to_datetime(timestamps) <= test_end_date)
        
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]
        
        if len(train_idx) > 100 and len(val_idx) > 50 and len(test_idx) > 50:
            splits.append((train_idx, val_idx, test_idx))
    
    return splits


class AdvancedFeatureEngineer:
    """Advanced feature engineering for trading models."""
    
    def __init__(self, config: AdvancedTrainConfig):
        self.config = config
        self.feature_importance_ = None
        self.selected_features_ = None
        self.correlation_matrix_ = None
        
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Feature selection based on importance and correlation."""
        print(f"Starting feature selection from {X.shape[1]} features...")
        
        # Remove highly correlated features first
        if self.config.remove_correlated:
            X = self._remove_correlated_features(X)
        
        # Feature importance-based selection
        if self.config.feature_selection and self.config.max_features:
            X = self._importance_based_selection(X, y)
        
        self.selected_features_ = list(X.columns)
        print(f"Selected {len(self.selected_features_)} features")
        return X
    
    def _remove_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        corr_matrix = X.corr().abs()
        self.correlation_matrix_ = corr_matrix
        
        # Find pairs of highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.config.correlation_threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        # Remove one from each highly correlated pair
        features_to_remove = set()
        for feat1, feat2 in high_corr_pairs:
            if feat1 not in features_to_remove:
                features_to_remove.add(feat2)
        
        remaining_features = [col for col in X.columns if col not in features_to_remove]
        print(f"Removed {len(features_to_remove)} highly correlated features")
        
        return X[remaining_features]
    
    def _importance_based_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select features based on importance using LightGBM."""
        from lightgbm import LGBMRegressor, LGBMClassifier
        
        # Determine if classification or regression
        if y.dtype == 'object' or len(np.unique(y)) < 10:
            model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
            if y.dtype == 'object':
                y_encoded = LabelEncoder().fit_transform(y)
            else:
                y_encoded = y
        else:
            model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            y_encoded = y
        
        # Handle missing values
        X_imputed = SimpleImputer().fit_transform(X)
        
        # Fit model and get feature importance
        model.fit(X_imputed, y_encoded)
        importance = model.feature_importances_
        
        # Select top features
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.feature_importance_ = feature_importance
        
        top_features = feature_importance.head(self.config.max_features)['feature'].tolist()
        return X[top_features]


class AdvancedModelTrainer:
    """Train multiple models and create ensembles."""
    
    def __init__(self, config: AdvancedTrainConfig):
        self.config = config
        self.models = {}
        self.ensemble_model = None
        self.label_encoder = None
        
    def create_base_models(self, task_type: str) -> Dict[str, Any]:
        """Create base models based on configuration."""
        models = {}
        
        if 'lightgbm' in self.config.model_types:
            if task_type == 'classification':
                models['lightgbm'] = LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                )
            else:
                models['lightgbm'] = LGBMRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                )
        
        if 'xgboost' in self.config.model_types:
            if task_type == 'classification':
                models['xgboost'] = XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss'
                )
            else:
                models['xgboost'] = XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
        
        if 'random_forest' in self.config.model_types:
            if task_type == 'classification':
                models['random_forest'] = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                models['random_forest'] = RandomForestRegressor(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )
        
        return models
    
    def train_models(self, X_train, y_train, X_val, y_val, task_type: str) -> Dict[str, Any]:
        """Train individual models."""
        base_models = self.create_base_models(task_type)
        trained_models = {}
        val_scores = {}
        
        # Handle categorical labels
        if task_type == 'classification' and y_train.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_val_encoded = self.label_encoder.transform(y_val)
        else:
            y_train_encoded = y_train
            y_val_encoded = y_val
        
        # Create imputer pipeline for each model
        for name, model in base_models.items():
            print(f"Training {name}...")
            
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('model', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train_encoded)
            trained_models[name] = pipeline
            
            # Validate
            if task_type == 'classification':
                if hasattr(pipeline, 'predict_proba'):
                    val_pred_proba = pipeline.predict_proba(X_val)
                    if val_pred_proba.shape[1] == 2:
                        val_scores[name] = roc_auc_score(y_val_encoded, val_pred_proba[:, 1])
                    else:
                        val_pred = pipeline.predict(X_val)
                        val_scores[name] = (val_pred == y_val_encoded).mean()
                else:
                    val_pred = pipeline.predict(X_val)
                    val_scores[name] = (val_pred == y_val_encoded).mean()
            else:
                val_pred = pipeline.predict(X_val)
                val_scores[name] = r2_score(y_val_encoded, val_pred)
            
            print(f"  {name} validation score: {val_scores[name]:.4f}")
        
        self.models = trained_models
        return trained_models, val_scores
    
    def create_ensemble(self, trained_models: Dict, task_type: str) -> Any:
        """Create ensemble model from trained base models."""
        if len(trained_models) < 2:
            print("Not enough models for ensemble, using single best model")
            return list(trained_models.values())[0]
        
        print(f"Creating {self.config.ensemble_method} ensemble...")
        
        if self.config.ensemble_method == 'voting':
            if task_type == 'classification':
                ensemble = VotingClassifier(
                    estimators=[(name, model) for name, model in trained_models.items()],
                    voting='soft'  # Use probabilities
                )
            else:
                ensemble = VotingRegressor(
                    estimators=[(name, model) for name, model in trained_models.items()]
                )
        else:
            # For now, default to voting. Could implement stacking/blending later
            if task_type == 'classification':
                ensemble = VotingClassifier(
                    estimators=[(name, model) for name, model in trained_models.items()],
                    voting='soft'
                )
            else:
                ensemble = VotingRegressor(
                    estimators=[(name, model) for name, model in trained_models.items()]
                )
        
        self.ensemble_model = ensemble
        return ensemble


def determine_task_type(y: pd.Series, target_type: str) -> str:
    """Determine if task is classification or regression."""
    if target_type in ['ranking', 'regime', 'earnings', 'reversion']:
        return 'classification' if y.dtype == 'object' or len(np.unique(y)) < 10 else 'regression'
    else:
        return 'regression' if y.dtype in ['float64', 'int64'] and len(np.unique(y)) > 10 else 'classification'


def advanced_train_main(config: AdvancedTrainConfig):
    """Main training function with advanced features."""
    print("="*60)
    print("ADVANCED ALGORITHMIC TRADING MODEL TRAINING")
    print("="*60)
    
    # Load data
    print("Loading data...")
    features = pd.read_parquet(config.features_path)
    labels = pd.read_parquet(config.labels_path)
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Get target variable
    if config.label_column not in labels.columns:
        available_cols = list(labels.columns)
        print(f"Label column '{config.label_column}' not found.")
        print(f"Available columns: {available_cols}")
        
        # Try to find a suitable column
        for col in available_cols:
            if 'rank' in col.lower() or 'label' in col.lower():
                config.label_column = col
                print(f"Using column: {col}")
                break
        else:
            raise ValueError(f"Could not find suitable label column")
    
    # Align data
    common_index = features.index.intersection(labels.index)
    X = features.loc[common_index]
    y = labels.loc[common_index, config.label_column]
    
    # Remove rows with missing targets
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]
    
    print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {y.value_counts().head()}")
    
    # Determine task type
    task_type = determine_task_type(y, config.prediction_target)
    print(f"Task type: {task_type}")
    
    # Feature engineering
    print("\nFeature Engineering...")
    feature_engineer = AdvancedFeatureEngineer(config)
    X_selected = feature_engineer.select_features(X, y)
    
    # Time-based splitting
    print("\nCreating time series splits...")
    if isinstance(X_selected.index, pd.MultiIndex):
        timestamps = X_selected.index.get_level_values(0)  # Assuming first level is timestamp
    else:
        timestamps = X_selected.index
    
    splits = create_time_series_splits(
        timestamps, 
        n_splits=config.n_cv_folds,
        test_frac=config.test_frac,
        val_frac=config.val_frac,
        embargo_days=config.embargo_days
    )
    
    if not splits:
        raise ValueError("Could not create valid time series splits")
    
    print(f"Created {len(splits)} time series splits")
    
    # Cross-validation training
    print("\nCross-Validation Training...")
    cv_scores = []
    all_models = []
    
    for i, (train_idx, val_idx, test_idx) in enumerate(splits):
        print(f"\nFold {i+1}/{len(splits)}")
        
        X_train = X_selected.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X_selected.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        # Train models for this fold
        trainer = AdvancedModelTrainer(config)
        trained_models, val_scores = trainer.train_models(X_train, y_train, X_val, y_val, task_type)
        
        # Create ensemble if requested
        if config.use_ensemble and len(trained_models) > 1:
            ensemble = trainer.create_ensemble(trained_models, task_type)
            # Note: ensemble needs to be fit separately since it's a new object
            if task_type == 'classification' and trainer.label_encoder:
                y_train_encoded = trainer.label_encoder.transform(y_train)
                y_val_encoded = trainer.label_encoder.transform(y_val)
            else:
                y_train_encoded = y_train
                y_val_encoded = y_val
            
            ensemble.fit(X_train, y_train_encoded)
            
            if task_type == 'classification':
                if hasattr(ensemble, 'predict_proba'):
                    ensemble_pred = ensemble.predict_proba(X_val)[:, 1] if ensemble.predict_proba(X_val).shape[1] == 2 else ensemble.predict(X_val)
                    ensemble_score = roc_auc_score(y_val_encoded, ensemble_pred) if ensemble.predict_proba(X_val).shape[1] == 2 else (ensemble.predict(X_val) == y_val_encoded).mean()
                else:
                    ensemble_score = (ensemble.predict(X_val) == y_val_encoded).mean()
            else:
                ensemble_score = r2_score(y_val_encoded, ensemble.predict(X_val))
            
            print(f"  Ensemble validation score: {ensemble_score:.4f}")
            val_scores['ensemble'] = ensemble_score
            trained_models['ensemble'] = ensemble
        
        cv_scores.append(val_scores)
        all_models.append(trained_models)
    
    # Select best model across all folds
    print("\nModel Selection...")
    avg_scores = {}
    for model_name in all_models[0].keys():
        scores = [fold_scores.get(model_name, 0) for fold_scores in cv_scores]
        avg_scores[model_name] = np.mean(scores)
        print(f"  {model_name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    best_model_name = max(avg_scores, key=avg_scores.get)
    print(f"\nBest model: {best_model_name} (score: {avg_scores[best_model_name]:.4f})")
    
    # Train final model on all data
    print(f"\nTraining final {best_model_name} model on all data...")
    final_trainer = AdvancedModelTrainer(config)
    
    # Split for final training (keep some data for testing)
    n_samples = len(X_selected)
    train_size = int(n_samples * (1 - config.test_frac))
    
    X_final_train = X_selected.iloc[:train_size]
    y_final_train = y.iloc[:train_size]
    X_final_test = X_selected.iloc[train_size:]
    y_final_test = y.iloc[train_size:]
    
    if best_model_name == 'ensemble':
        final_models, _ = final_trainer.train_models(X_final_train, y_final_train, X_final_test, y_final_test, task_type)
        final_model = final_trainer.create_ensemble(final_models, task_type)
        
        if task_type == 'classification' and final_trainer.label_encoder:
            y_final_train_encoded = final_trainer.label_encoder.transform(y_final_train)
        else:
            y_final_train_encoded = y_final_train
        
        final_model.fit(X_final_train, y_final_train_encoded)
    else:
        final_models, _ = final_trainer.train_models(X_final_train, y_final_train, X_final_test, y_final_test, task_type)
        final_model = final_models[best_model_name]
    
    # Final evaluation
    print("\nFinal Evaluation...")
    if task_type == 'classification':
        if final_trainer.label_encoder:
            y_test_encoded = final_trainer.label_encoder.transform(y_final_test)
        else:
            y_test_encoded = y_final_test
            
        if hasattr(final_model, 'predict_proba'):
            test_pred_proba = final_model.predict_proba(X_final_test)
            if test_pred_proba.shape[1] == 2:
                test_score = roc_auc_score(y_test_encoded, test_pred_proba[:, 1])
                print(f"Final Test AUC: {test_score:.4f}")
            else:
                test_pred = final_model.predict(X_final_test)
                test_score = (test_pred == y_test_encoded).mean()
                print(f"Final Test Accuracy: {test_score:.4f}")
        else:
            test_pred = final_model.predict(X_final_test)
            test_score = (test_pred == y_test_encoded).mean()
            print(f"Final Test Accuracy: {test_score:.4f}")
    else:
        test_pred = final_model.predict(X_final_test)
        test_score = r2_score(y_final_test, test_pred)
        print(f"Final Test R²: {test_score:.4f}")
    
    # Save model and metadata
    os.makedirs(config.model_output_dir, exist_ok=True)
    
    model_path = os.path.join(config.model_output_dir, f"{config.experiment_name}_model.pkl")
    meta_path = os.path.join(config.model_output_dir, f"{config.experiment_name}_meta.json")
    
    joblib.dump(final_model, model_path)
    
    # Create metadata
    metadata = {
        'experiment_name': config.experiment_name,
        'prediction_target': config.prediction_target,
        'task_type': task_type,
        'best_model_type': best_model_name,
        'feature_names': feature_engineer.selected_features_,
        'n_features': len(feature_engineer.selected_features_),
        'label_column': config.label_column,
        'cv_scores': {model: scores for model, scores in avg_scores.items()},
        'final_test_score': test_score,
        'label_encoder_classes': final_trainer.label_encoder.classes_.tolist() if final_trainer.label_encoder else None,
        'model_config': asdict(config)
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Metadata saved to: {meta_path}")
    
    return final_model, metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Advanced Trading Model Training")
    parser.add_argument("--features_path", required=True, help="Path to features parquet file")
    parser.add_argument("--labels_path", required=True, help="Path to labels parquet file")
    parser.add_argument("--prediction_target", default="ranking", choices=["ranking", "regime", "reversion", "earnings"], help="Type of prediction target")
    parser.add_argument("--label_column", default="rank_10d", help="Column name for target variable")
    parser.add_argument("--model_types", nargs="+", default=["lightgbm", "xgboost", "random_forest"], help="Model types to use")
    parser.add_argument("--use_ensemble", action="store_true", help="Create ensemble model")
    parser.add_argument("--experiment_name", default="advanced_trading_model", help="Experiment name for output files")
    parser.add_argument("--test_frac", type=float, default=0.2, help="Fraction of data for testing")
    parser.add_argument("--max_features", type=int, default=100, help="Maximum number of features to select")
    parser.add_argument("--output_dir", default="models", help="Output directory for models")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    config = AdvancedTrainConfig(
        features_path=args.features_path,
        labels_path=args.labels_path,
        prediction_target=args.prediction_target,
        label_column=args.label_column,
        model_types=args.model_types,
        use_ensemble=args.use_ensemble,
        experiment_name=args.experiment_name,
        test_frac=args.test_frac,
        max_features=args.max_features,
        model_output_dir=args.output_dir
    )
    
    final_model, metadata = advanced_train_main(config)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model: {metadata['best_model_type']}")
    print(f"Final score: {metadata['final_test_score']:.4f}")
    print(f"Features selected: {metadata['n_features']}")
    print("="*60)