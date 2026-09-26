import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.models.gbm import MatchingGBM
from src.threshold import format_predictions_by_threshold, optimize_threshold, write_matching_results_tsv


def test_matching_gbm_fit_predict(tmp_path):
    # Create synthetic dataset with clear signal
    np.random.seed(42)
    n_samples = 200
    X = pd.DataFrame({
        "name_exact_match": np.random.choice([0.0, 1.0], size=n_samples, p=[0.8, 0.2]),
        "name_levenshtein_ratio": np.random.uniform(0.0, 1.0, size=n_samples),
        "addr_numeric_exact_match": np.random.choice([0.0, 1.0], size=n_samples, p=[0.7, 0.3])
    })
    # Labels correlate strongly with exact match and high levenshtein
    y = ((X["name_exact_match"] == 1.0) | (X["name_levenshtein_ratio"] > 0.8)).astype(float)

    model = MatchingGBM()
    importances = model.fit(X, y, num_boost_round=20)
    
    assert len(importances) == 3
    assert "name_levenshtein_ratio" in importances or "name_exact_match" in importances
    
    probs = model.predict_proba(X)
    assert len(probs) == n_samples
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # Test serialization
    save_path = tmp_path / "model.joblib"
    model.save(save_path)
    loaded_model = MatchingGBM.load(save_path)
    loaded_probs = loaded_model.predict_proba(X)
    np.testing.assert_allclose(probs, loaded_probs)


def test_threshold_optimization():
    s1_series = pd.Series(["S1-1", "S1-1", "S1-2", "S1-3"])
    cand_series = pd.Series(["S2-1", "S2-2", "S3-1", "S2-3"])
    probs = np.array([0.95, 0.30, 0.15, 0.85])
    
    gt = {
        "S1-1": ["S2-1"], # S2-1 is true, S2-2 is distractor
        "S1-2": [],       # singleton
        "S1-3": ["S2-3"]  # true match
    }
    all_s1 = ["S1-1", "S1-2", "S1-3"]

    best_tau, metrics, sweep_df = optimize_threshold(
        s1_series, cand_series, probs, gt, all_s1, threshold_grid=np.array([0.20, 0.50, 0.80, 0.90])
    )
    
    # At threshold 0.50 or 0.80, S1-1 gets S2-1 (correct), S1-2 gets [] (correct singleton), S1-3 gets S2-3 (correct) -> 1.0 F0.5
    assert best_tau in (0.50, 0.80)
    assert metrics["macro_f05"] == 1.0
    assert metrics["singleton_accuracy"] == 1.0


def test_write_matching_results_tsv(tmp_path):
    preds = {
        "S1-001": ["S2-001", "S3-002"],
        "S1-002": [],
        "S1-003": ["S2-003"]
    }
    tsv_file = tmp_path / "matching_results.tsv"
    write_matching_results_tsv(preds, tsv_file, all_s1_ids=["S1-001", "S1-002", "S1-003"])

    lines = tsv_file.read_text(encoding="utf-8").strip().split("\n")
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert lines[1] == "S1-001\tS2-001,S3-002"
    assert lines[2] == "S1-002\t"
    assert lines[3] == "S1-003\tS2-003"
