"""
Phase 5 Training & Validation Evaluation Pipeline.
Trains LightGBM Matching Classifier on generated feature matrices, sweeps decision thresholds
for optimal Macro-F_0.5, and reports detailed diagnostics (singleton vs non-singleton, precision vs recall).
"""

import sys
from pathlib import Path

# Add package root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.utils import load_config, set_seed, Timer
from src.data_io import load_ground_truth
from src.models.gbm import MatchingGBM
from src.threshold import optimize_threshold, format_predictions_by_threshold
from src.evaluate import evaluate_predictions


def run_training_and_evaluation():
    config = load_config()
    set_seed(config["seed"])
    paths = config["paths"]
    cache_dir = Path(paths["cache_dir"])

    print("=" * 65)
    print(" Amazon ML Challenge 2026 — Phase 5: Matching Model & Calibration ")
    print("=" * 65)

    feat_path = cache_dir / "val_features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature matrix {feat_path} does not exist! Run build_training_features.py first.")

    with Timer("Loading Feature Matrix"):
        df = pd.read_parquet(feat_path)
        print(f"Loaded {len(df):,} candidate pair rows.")

    gt_df = load_ground_truth(paths["train_gt"], parse_matches=True)
    all_s1_ids = df["s1_id"].unique().tolist()
    
    # Build ground truth dict for scoped S1 entities
    gt_scoped = gt_df[gt_df["source1_entity_id"].isin(all_s1_ids)]
    gt_dict = {}
    for _, r in gt_scoped.iterrows():
        gt_dict[r["source1_entity_id"]] = r["match_list"] if "match_list" in r else r["matched_entity_ids"]
    # Ensure any singleton not in gt_df has empty list
    for s1 in all_s1_ids:
        if s1 not in gt_dict:
            gt_dict[s1] = []

    # Entity-Stratified Group Split (80% train / 20% holdout val)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=config["seed"])
    train_idx, val_idx = next(gss.split(df, groups=df["s1_id"]))

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    feature_cols = [c for c in df.columns if c not in ("s1_id", "cand_id", "label")]
    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    X_val = val_df[feature_cols]
    y_val = val_df["label"]

    val_s1_ids = val_df["s1_id"].unique().tolist()

    print(f"\n--- Split Summary ---")
    print(f"Train: {len(X_train):,} pairs across {train_df['s1_id'].nunique():,} S1 entities | Positives = {y_train.sum():,}")
    print(f"Val:   {len(X_val):,} pairs across {len(val_s1_ids):,} S1 entities | Positives = {y_val.sum():,}")

    # Train LightGBM
    with Timer("Training LightGBM Classifier"):
        model = MatchingGBM()
        importances = model.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            num_boost_round=300,
            early_stopping_rounds=30
        )

    # Feature Importance Analysis
    print("\n--- Top 10 Most Predictive Features ---")
    for i, (feat, imp) in enumerate(list(importances.items())[:10], 1):
        print(f"  {i:2d}. {feat:<28} : {imp:10.2f}")

    # Probability Predictions on Holdout Validation Set
    val_probs = model.predict_proba(X_val)

    # Threshold Optimization for Macro-F_0.5
    with Timer("Calibrating Optimal Decision Threshold"):
        best_tau, best_metrics, sweep_df = optimize_threshold(
            s1_series=val_df["s1_id"],
            cand_series=val_df["cand_id"],
            probs=val_probs,
            ground_truth=gt_dict,
            all_s1_ids=val_s1_ids,
            threshold_grid=np.linspace(0.10, 0.95, 86),
            beta=0.5
        )

    print("\n" + "=" * 65)
    print(f" ⭐ Optimal Decision Threshold: tau = {best_tau:.3f} ⭐ ")
    print("=" * 65)
    print(f" Validation Macro F_0.5 Score  : {best_metrics['macro_f05'] * 100:6.2f}%")
    print(f" Macro Precision (2x weight)   : {best_metrics['macro_precision'] * 100:6.2f}%")
    print(f" Macro Recall                  : {best_metrics['macro_recall'] * 100:6.2f}%")
    print(f" Singleton Accuracy (Strict)   : {best_metrics['singleton_accuracy'] * 100:6.2f}%")
    print(f" Non-Singleton F_0.5           : {best_metrics['non_singleton_f05'] * 100:6.2f}%")
    print(f" Total Scored Validation S1s   : {best_metrics['num_entities']:,} (Singletons = {best_metrics['num_singletons']:,})")
    print("=" * 65)

    # Save trained model and threshold
    model_save_path = cache_dir / "lgbm_model.joblib"
    model.save(model_save_path)
    print(f"\nSaved model to: {model_save_path}")

    return best_tau, best_metrics


if __name__ == "__main__":
    run_training_and_evaluation()
