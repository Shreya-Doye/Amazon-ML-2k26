"""
Official Macro-Averaged F_0.5 Evaluation Metric for Business Entity Resolution.
Implements the exact evaluation criteria specified in the ML Challenge 2026 problem statement.
"""

from typing import Dict, List, Set, Union, Optional
import pandas as pd
import numpy as np


def compute_fbeta_per_entity(
    predicted: Union[Set[str], List[str], str],
    ground_truth: Union[Set[str], List[str], str],
    beta: float = 0.5
) -> float:
    """
    Computes F_beta score for a single Source 1 entity.
    
    Rules:
    - If ground truth is empty (singleton):
        - Returns 1.0 if predicted is empty (correct singleton identification)
        - Returns 0.0 if predicted is non-empty (false merge on singleton)
    - If ground truth is non-empty:
        - Returns 0.0 if predicted is empty (missed match)
        - Otherwise calculates standard F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
    """
    if isinstance(predicted, str):
        pred_set = {x.strip() for x in predicted.split(",") if x.strip()}
    elif isinstance(predicted, (set, list, tuple)):
        pred_set = set(predicted)
    else:
        pred_set = set()

    if isinstance(ground_truth, str):
        gt_set = {x.strip() for x in ground_truth.split(",") if x.strip()}
    elif isinstance(ground_truth, (set, list, tuple)):
        gt_set = set(ground_truth)
    else:
        gt_set = set()

    # Singleton case
    if len(gt_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0

    # Non-singleton with empty prediction
    if len(pred_set) == 0:
        return 0.0

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    beta_sq = beta ** 2
    denom = (beta_sq * precision + recall)
    if denom == 0.0:
        return 0.0

    return (1.0 + beta_sq) * precision * recall / denom


def evaluate_predictions(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    all_s1_ids: Optional[List[str]] = None,
    beta: float = 0.5
) -> Dict[str, float]:
    """
    Computes macro-averaged F_0.5 and auxiliary diagnostics across all Source 1 entities.
    
    Args:
        predictions: Dict mapping s1_id -> list/set of predicted matching entity IDs
        ground_truth: Dict mapping s1_id -> list/set of true matching entity IDs
        all_s1_ids: Optional explicit list of all S1 IDs in evaluation set. If None, uses union of keys.
        beta: Beta parameter for F-score (default 0.5)
        
    Returns:
        Dict with macro_f05, precision, recall, singleton accuracy, non-singleton F_0.5, etc.
    """
    if all_s1_ids is None:
        target_ids = list(set(predictions.keys()) | set(ground_truth.keys()))
    else:
        target_ids = list(all_s1_ids)

    if not target_ids:
        return {
            "macro_f05": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "singleton_accuracy": 0.0,
            "non_singleton_f05": 0.0,
            "num_entities": 0,
            "num_singletons": 0
        }

    scores = []
    precisions = []
    recalls = []
    singleton_scores = []
    non_singleton_scores = []

    for s1_id in target_ids:
        pred_val = predictions.get(s1_id, [])
        gt_val = ground_truth.get(s1_id, [])

        if isinstance(pred_val, str):
            pred_set = {x.strip() for x in pred_val.split(",") if x.strip()}
        elif isinstance(pred_val, (set, list, tuple)):
            pred_set = set(pred_val)
        else:
            pred_set = set()

        if isinstance(gt_val, str):
            gt_set = {x.strip() for x in gt_val.split(",") if x.strip()}
        elif isinstance(gt_val, (set, list, tuple)):
            gt_set = set(gt_val)
        else:
            gt_set = set()

        f_score = compute_fbeta_per_entity(pred_set, gt_set, beta=beta)
        scores.append(f_score)

        if len(gt_set) == 0:
            singleton_scores.append(f_score)
            precisions.append(1.0 if len(pred_set) == 0 else 0.0)
            recalls.append(1.0 if len(pred_set) == 0 else 0.0)
        else:
            non_singleton_scores.append(f_score)
            tp = len(pred_set & gt_set)
            p = tp / len(pred_set) if len(pred_set) > 0 else 0.0
            r = tp / len(gt_set) if len(gt_set) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)

    return {
        "macro_f05": float(np.mean(scores)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "singleton_accuracy": float(np.mean(singleton_scores)) if singleton_scores else 0.0,
        "non_singleton_f05": float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0,
        "num_entities": len(target_ids),
        "num_singletons": len(singleton_scores)
    }


def evaluate_tsv(
    pred_tsv_path: str,
    gt_tsv_path: str,
    val_s1_ids: Optional[List[str]] = None,
    beta: float = 0.5
) -> Dict[str, float]:
    """Evaluates prediction TSV directly against ground truth TSV."""
    pred_df = pd.read_csv(pred_tsv_path, sep="\t", dtype=str, keep_default_na=False)
    gt_df = pd.read_csv(gt_tsv_path, sep="\t", dtype=str, keep_default_na=False)

    def parse_matches(df, id_col, match_col):
        res = {}
        for _, row in df.iterrows():
            s1 = row[id_col]
            raw = str(row[match_col]).strip()
            res[s1] = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
        return res

    pred_dict = parse_matches(pred_df, "source1_entity_id", "matched_entity_ids")
    gt_dict = parse_matches(gt_df, "source1_entity_id", "matched_entity_ids")

    return evaluate_predictions(pred_dict, gt_dict, all_s1_ids=val_s1_ids, beta=beta)
