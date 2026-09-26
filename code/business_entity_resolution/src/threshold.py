"""
Threshold Calibration & Submission Generation for Entity Resolution.
Finds the global decision threshold tau that maximizes the official Macro-Averaged F_0.5 metric,
and formats predictions into the final matching_results.tsv.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from src.evaluate import evaluate_predictions


def format_predictions_by_threshold(
    s1_series: pd.Series,
    cand_series: pd.Series,
    probs: np.ndarray,
    threshold: float,
    all_s1_ids: List[str]
) -> Dict[str, List[str]]:
    """
    Groups pairs where prob >= threshold per S1 entity, ensuring all target S1 IDs are present.
    """
    # Filter by threshold
    mask = probs >= threshold
    filtered_s1 = s1_series[mask]
    filtered_cands = cand_series[mask]

    # Initialize all entities as singletons (empty list)
    predictions: Dict[str, List[str]] = {s1_id: [] for s1_id in all_s1_ids}

    # Group matches
    for s1, cand in zip(filtered_s1, filtered_cands):
        if s1 in predictions:
            predictions[s1].append(cand)
        else:
            predictions[s1] = [cand]

    return predictions


def optimize_threshold(
    s1_series: pd.Series,
    cand_series: pd.Series,
    probs: np.ndarray,
    ground_truth: Dict[str, List[str]],
    all_s1_ids: List[str],
    threshold_grid: Optional[np.ndarray] = None,
    beta: float = 0.5
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """
    Performs grid search over candidate decision thresholds to maximize Macro-F_beta.
    
    Returns:
        best_threshold: Optimal decision threshold
        best_metrics: Performance dictionary at optimal threshold
        sweep_df: DataFrame of all threshold results
    """
    if threshold_grid is None:
        threshold_grid = np.linspace(0.10, 0.95, 86)

    records = []
    best_fbeta = -1.0
    best_tau = 0.50
    best_metrics = {}

    for tau in threshold_grid:
        preds = format_predictions_by_threshold(s1_series, cand_series, probs, tau, all_s1_ids)
        metrics = evaluate_predictions(preds, ground_truth, all_s1_ids=all_s1_ids, beta=beta)
        f_score = metrics["macro_f05"]

        records.append({
            "threshold": float(tau),
            "macro_f05": f_score,
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "singleton_acc": metrics["singleton_accuracy"],
            "non_singleton_f05": metrics["non_singleton_f05"]
        })

        if f_score > best_fbeta:
            best_fbeta = f_score
            best_tau = float(tau)
            best_metrics = metrics

    sweep_df = pd.DataFrame(records)
    return best_tau, best_metrics, sweep_df


def write_matching_results_tsv(
    predictions: Dict[str, List[str]],
    output_path: Path,
    all_s1_ids: Optional[List[str]] = None
):
    """
    Writes predictions to matching_results.tsv strictly following competition guidelines:
    - Tab-separated
    - Header: source1_entity_id\tmatched_entity_ids
    - Comma-separated list with no spaces, quotes, or trailing separators
    - Every S1 entity present exactly once
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_ids = all_s1_ids if all_s1_ids is not None else sorted(predictions.keys())

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in target_ids:
            matches = predictions.get(s1_id, [])
            # Deduplicate preserving order
            seen = set()
            unique_matches = []
            for m in matches:
                if m not in seen:
                    seen.add(m)
                    unique_matches.append(m)
            match_str = ",".join(unique_matches)
            f.write(f"{s1_id}\t{match_str}\n")
