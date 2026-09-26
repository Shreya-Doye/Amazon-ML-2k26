import pytest
from src.evaluate import compute_fbeta_per_entity, evaluate_predictions


def test_readme_worked_example():
    """
    Direct unit test against the problem statement worked example:
    - Model predicts S1-00001 matches [S2-00047, S2-00193, S3-00812]
    - Ground truth says S1-00001 matches [S2-00047, S3-00812]
    - Precision = 2/3, Recall = 1.0
    - F_0.5 = (1.25 * 0.667 * 1.0) / (0.25 * 0.667 + 1.0) = 0.714
    """
    predicted = ["S2-00047", "S2-00193", "S3-00812"]
    ground_truth = ["S2-00047", "S3-00812"]
    
    score = compute_fbeta_per_entity(predicted, ground_truth, beta=0.5)
    assert round(score, 3) == 0.714, f"Expected 0.714, got {score}"


def test_singleton_scoring():
    # Singleton correctly predicted as empty -> 1.0
    assert compute_fbeta_per_entity([], [], beta=0.5) == 1.0
    
    # Singleton incorrectly matched -> 0.0
    assert compute_fbeta_per_entity(["S2-99999"], [], beta=0.5) == 0.0


def test_non_singleton_missed():
    # Non-singleton missed entirely -> 0.0
    assert compute_fbeta_per_entity([], ["S2-00001"], beta=0.5) == 0.0


def test_perfect_match():
    # Perfect match -> 1.0
    assert compute_fbeta_per_entity(["S2-1", "S3-2"], ["S2-1", "S3-2"], beta=0.5) == 1.0


def test_macro_evaluation_aggregation():
    gt = {
        "S1-1": ["S2-1", "S3-1"], # Readme example -> 0.7142857...
        "S1-2": [],                # Correct singleton -> 1.0
        "S1-3": [],                # False merge singleton -> 0.0
        "S1-4": ["S2-4"]           # Perfect match -> 1.0
    }
    preds = {
        "S1-1": ["S2-1", "S2-9", "S3-1"],
        "S1-2": [],
        "S1-3": ["S2-3"],
        "S1-4": ["S2-4"]
    }
    
    results = evaluate_predictions(preds, gt, beta=0.5)
    assert results["num_entities"] == 4
    assert results["num_singletons"] == 2
    assert results["singleton_accuracy"] == 0.5  # 1 out of 2 correct
    
    expected_macro = (0.7142857142857143 + 1.0 + 0.0 + 1.0) / 4.0
    assert abs(results["macro_f05"] - expected_macro) < 1e-5
