"""
Data I/O Module for Amazon ML Challenge 2026 Entity Resolution pipeline.
Handles robust TSV ingestion with strict string types, schema assertions,
stratified train/validation splitting, and ground truth extraction.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
EXPECTED_PREFIXES = {
    "source1": "S1-",
    "source2": "S2-",
    "source3": "S3-",
    "s1": "S1-",
    "s2": "S2-",
    "s3": "S3-",
}


def validate_source_schema(df: pd.DataFrame, source_hint: Optional[str] = None) -> None:
    """
    Validate that source DataFrame contains required columns and correct prefix format.
    Raises ValueError on schema or format violation.
    """
    missing_cols = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema violation: missing columns {missing_cols} in source DataFrame.")

    if len(df) > 0 and source_hint:
        hint_key = source_hint.lower()
        expected_prefix = None
        for k, p in EXPECTED_PREFIXES.items():
            if k in hint_key:
                expected_prefix = p
                break

        if expected_prefix:
            sample_ids = df["entity_id"].dropna().iloc[:100]
            invalid = [eid for eid in sample_ids if not str(eid).startswith(expected_prefix)]
            if invalid:
                raise ValueError(
                    f"Entity ID format violation for {source_hint}: expected prefix '{expected_prefix}', "
                    f"found invalid samples like '{invalid[:3]}'."
                )


def validate_ground_truth_schema(df: pd.DataFrame) -> None:
    """Validate ground truth DataFrame columns and structure."""
    missing_cols = [c for c in GT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema violation: missing columns {missing_cols} in Ground Truth DataFrame.")


def load_source(
    path: Union[str, Path],
    assert_schema: bool = True,
    nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load a source file (Source 1, Source 2, or Source 3) with strict tab separation and string types.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found at: {path}")

    # Tabs are mandatory to avoid comma-based collapse
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=nrows
    )

    if assert_schema:
        validate_source_schema(df, source_hint=path.stem)

    return df


def load_ground_truth(
    path: Union[str, Path],
    parse_matches: bool = False,
    nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load ground truth matching file with strict tab separation.
    Optionally parses comma-separated matched_entity_ids into Python list of strings.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found at: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=nrows
    )

    validate_ground_truth_schema(df)

    if parse_matches:
        df["match_list"] = df["matched_entity_ids"].apply(
            lambda x: [m.strip() for m in x.split(",") if m.strip()] if isinstance(x, str) and x.strip() else []
        )
        df["is_singleton"] = df["match_list"].apply(lambda l: len(l) == 0)
        df["num_matches"] = df["match_list"].apply(len)

    return df


def create_stratified_validation_split(
    s1_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    val_ratio: float = 0.20,
    seed: int = 42,
    save_path: Optional[Union[str, Path]] = None
) -> Dict[str, Union[List[str], Dict[str, Any]]]:
    """
    Create a stratified train/validation split strictly by S1 entity ID.
    Stratified on (country, is_singleton) to preserve class distributions.
    """
    # Merge S1 metadata with GT match info
    if "is_singleton" not in gt_df.columns or "match_list" not in gt_df.columns:
        gt_df = gt_df.copy()
        gt_df["match_list"] = gt_df["matched_entity_ids"].apply(
            lambda x: [m.strip() for m in x.split(",") if m.strip()] if isinstance(x, str) and x.strip() else []
        )
        gt_df["is_singleton"] = gt_df["match_list"].apply(lambda l: len(l) == 0)

    merged = s1_df[["entity_id", "country"]].merge(
        gt_df[["source1_entity_id", "is_singleton"]],
        left_on="entity_id",
        right_on="source1_entity_id",
        how="inner"
    )

    # Stratification label: e.g. "US_singleton", "US_multi", "India_singleton", "India_multi"
    country_clean = merged["country"].str.strip().str.lower()
    merged["strata"] = country_clean + "_" + merged["is_singleton"].astype(str)

    # Perform stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(merged, merged["strata"]))

    train_ids = sorted(merged.iloc[train_idx]["entity_id"].tolist())
    val_ids = sorted(merged.iloc[val_idx]["entity_id"].tolist())

    # Verify zero leakage
    train_set = set(train_ids)
    val_set = set(val_ids)
    assert len(train_set.intersection(val_set)) == 0, "Data leakage detected: train and val IDs overlap!"
    assert len(train_ids) + len(val_ids) == len(merged), "Mismatch in total split IDs count!"

    # Calculate statistics
    train_df_sub = merged[merged["entity_id"].isin(train_set)]
    val_df_sub = merged[merged["entity_id"].isin(val_set)]

    stats = {
        "total_s1_entities": len(merged),
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "val_ratio_actual": len(val_ids) / len(merged),
        "train_singleton_ratio": float(train_df_sub["is_singleton"].mean()),
        "val_singleton_ratio": float(val_df_sub["is_singleton"].mean()),
        "train_countries": train_df_sub["country"].value_counts().to_dict(),
        "val_countries": val_df_sub["country"].value_counts().to_dict(),
    }

    result = {
        "seed": seed,
        "train_s1_ids": train_ids,
        "val_s1_ids": val_ids,
        "stats": stats
    }

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


def load_validation_split(split_path: Union[str, Path]) -> Dict[str, Any]:
    """Load previously saved validation split JSON."""
    split_path = Path(split_path)
    if not split_path.exists():
        raise FileNotFoundError(f"Validation split file not found at: {split_path}")
    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_scoped_matched_ids(s1_ids: Set[str], gt_df: pd.DataFrame) -> Set[str]:
    """
    Extract all S2 and S3 candidate entity IDs matched by a given set of S1 IDs.
    """
    gt_sub = gt_df[gt_df["source1_entity_id"].isin(s1_ids)]
    matched_ids = set()
    for matches in gt_sub["matched_entity_ids"]:
        if isinstance(matches, str) and matches.strip():
            for m in matches.split(","):
                m_clean = m.strip()
                if m_clean:
                    matched_ids.add(m_clean)
    return matched_ids
