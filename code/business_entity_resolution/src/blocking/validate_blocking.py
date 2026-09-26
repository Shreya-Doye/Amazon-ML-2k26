"""
Blocking Validation & Recall Ceiling Evaluation Module.
Computes recall ceiling, reduction ratio, candidate set statistics,
and verifies candidate format compliance.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd


def evaluate_blocking_recall(
    candidates_by_s1: Dict[str, Set[str]],
    gt_df: pd.DataFrame,
    s1_ids_subset: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate blocking recall against ground truth matching data.
    """
    if "match_list" not in gt_df.columns:
        gt_df = gt_df.copy()
        gt_df["match_list"] = gt_df["matched_entity_ids"].apply(
            lambda x: [m.strip() for m in x.split(",") if m.strip()] if isinstance(x, str) and x.strip() else []
        )

    total_matches = 0
    recovered_matches = 0
    total_non_singletons = 0
    perfect_s1_recall = 0

    evaluated_s1_count = 0
    cand_counts = []

    for row in gt_df.itertuples():
        s1_id = row.source1_entity_id
        if s1_ids_subset is not None and s1_id not in s1_ids_subset:
            continue

        evaluated_s1_count += 1
        true_matches = set(row.match_list)
        cands = candidates_by_s1.get(s1_id, set())
        cand_counts.append(len(cands))

        if len(true_matches) > 0:
            total_non_singletons += 1
            total_matches += len(true_matches)
            recovered = len(true_matches.intersection(cands))
            recovered_matches += recovered
            if recovered == len(true_matches):
                perfect_s1_recall += 1

    recall_ceiling = recovered_matches / max(1, total_matches)
    s1_coverage_rate = perfect_s1_recall / max(1, total_non_singletons)

    cand_array = np.array(cand_counts) if cand_counts else np.array([0])

    stats = {
        "evaluated_s1_entities": evaluated_s1_count,
        "total_true_match_links": total_matches,
        "recovered_match_links": recovered_matches,
        "recall_ceiling": float(recall_ceiling),
        "recall_ceiling_pct": float(recall_ceiling * 100),
        "s1_perfect_coverage_rate": float(s1_coverage_rate),
        "mean_candidates_per_s1": float(np.mean(cand_array)),
        "median_candidates_per_s1": float(np.median(cand_array)),
        "p95_candidates_per_s1": float(np.percentile(cand_array, 95)),
        "max_candidates_per_s1": int(np.max(cand_array)),
        "min_candidates_per_s1": int(np.min(cand_array)),
    }

    return stats
