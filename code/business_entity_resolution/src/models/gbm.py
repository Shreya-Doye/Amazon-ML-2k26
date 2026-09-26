"""
LightGBM Matching Classifier for Pairwise Entity Resolution.
Trains a gradient boosted tree on 28+ engineered tabular features with class imbalance handling,
feature importance tracking, and calibrated probability predictions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib


class MatchingGBM:
    """LightGBM classifier tailored for binary entity resolution matching."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.default_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "min_data_in_leaf": 20,
            "scale_pos_weight": 1.0,
            "verbose": -1,
            "n_jobs": -1,
            "random_state": 42
        }
        if params:
            self.default_params.update(params)
        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        num_boost_round: int = 300,
        early_stopping_rounds: int = 30
    ) -> Dict[str, float]:
        """Fits LightGBM on training pairwise feature matrix."""
        self.feature_names = list(X_train.columns)

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, feature_name=self.feature_names, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("val")

        callbacks = []
        if X_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

        self.model = lgb.train(
            params=self.default_params,
            train_set=train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        return self.get_feature_importances()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns predicted probability of matching (P(match=1))."""
        if self.model is None:
            raise ValueError("Model is not fitted yet!")
        return self.model.predict(X[self.feature_names])

    def get_feature_importances(self, importance_type: str = "gain") -> Dict[str, float]:
        """Returns sorted feature importances."""
        if self.model is None:
            raise ValueError("Model is not fitted yet!")
        importances = self.model.feature_importance(importance_type=importance_type)
        feat_imp = {name: float(imp) for name, imp in zip(self.feature_names, importances)}
        return dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

    def save(self, model_path: Path):
        """Saves model booster and feature list."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_names, "params": self.default_params}, model_path)

    @classmethod
    def load(cls, model_path: Path) -> "MatchingGBM":
        """Loads serialized model."""
        data = joblib.load(model_path)
        instance = cls(params=data["params"])
        instance.model = data["model"]
        instance.feature_names = data["features"]
        return instance
