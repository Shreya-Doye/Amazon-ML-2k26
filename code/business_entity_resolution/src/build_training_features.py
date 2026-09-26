"""
Feature Matrix Generation and Parquet Caching Pipeline.
Extracts pairwise feature vectors for training/validation candidate pairs,
joins ground truth labels, and caches matrices to parquet for Phase 5 models.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Set

# Add package root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import pandas as pd
from src.data_io import load_source, load_ground_truth, load_validation_split, get_scoped_matched_ids
from src.utils import load_config, set_seed, Timer
from src.normalize_name import NameNormalizer
from src.normalize_address import AddressNormalizer
from src.segmentation import WordSegmenter
from src.batch_normalize import process_dataframe
from src.blocking.country_bucket import partition_by_country
from src.blocking.union_blocking import MultiChannelBlocker
from src.features import FeatureExtractor, build_feature_matrix


def run_feature_pipeline(num_s1_sample: int = 5000):
    config = load_config()
    set_seed(config["seed"])
    paths = config["paths"]
    cache_dir = Path(paths["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" Amazon ML Challenge 2026 — Phase 4: Pairwise Feature Matrix ")
    print("=" * 60)

    # 1. Load Ground Truth and Validation Split
    gt_full = load_ground_truth(paths["train_gt"], parse_matches=True)
    val_split = load_validation_split(paths["val_split"])
    val_s1_set = set(val_split["val_s1_ids"][:num_s1_sample])

    print(f"Loaded validation split: evaluating {len(val_s1_set):,} validation S1 entities...")

    # 2. Load S1, S2, S3 records
    with Timer("Loading Source Records"):
        s1_full = load_source(paths["train_s1"])
        s2_full = load_source(paths["train_s2"])
        s3_full = load_source(paths["train_s3"])

        s1_val = s1_full[s1_full["entity_id"].isin(val_s1_set)].copy()
        gt_val = gt_full[gt_full["source1_entity_id"].isin(val_s1_set)].copy()

        # Find target S2/S3 matches + distractors
        target_cand_ids = get_scoped_matched_ids(val_s1_set, gt_val)
        s2_matches = s2_full[s2_full["entity_id"].isin(target_cand_ids)]
        s3_matches = s3_full[s3_full["entity_id"].isin(target_cand_ids)]
        
        # Add distractors
        s2_distractors = s2_full.sample(n=min(15000, len(s2_full)), random_state=42)
        s3_distractors = s3_full.sample(n=min(15000, len(s3_full)), random_state=42)

        cand_raw = pd.concat([s2_matches, s3_matches, s2_distractors, s3_distractors], ignore_index=True).drop_duplicates(subset=["entity_id"])

    print(f"[Validation Scope] S1: {len(s1_val):,} | Candidates (Matches + Distractors): {len(cand_raw):,}")

    # 3. Normalize Records
    with Timer("Normalizing Records"):
        segmenter = WordSegmenter()
        vocab_path = cache_dir / "vocab_freq.json"
        if vocab_path.exists():
            segmenter.load(vocab_path)
            
        name_norm = NameNormalizer(paths["legal_suffixes"], segmenter=segmenter)
        addr_norm = AddressNormalizer(paths["address_abbreviations"])

        s1_norm = process_dataframe(s1_val, name_norm, addr_norm)
        cand_norm = process_dataframe(cand_raw, name_norm, addr_norm)

    # 4. Multi-Channel Blocking
    with Timer("Executing Multi-Channel Blocking"):
        s1_by_c = partition_by_country(s1_norm)
        cand_by_c = partition_by_country(cand_norm)

        blocker = MultiChannelBlocker(
            df_cap_percentile=config["blocking"]["df_cap_percentile"],
            ngram_jaccard_threshold=config["blocking"]["ngram_jaccard_threshold"]
        )

        all_cands: Dict[str, Set[str]] = {}
        all_prov = {}

        for c_key, s1_grp in s1_by_c.items():
            cand_grp = cand_by_c.get(c_key, pd.DataFrame(columns=cand_norm.columns))
            cands, prov = blocker.block_country_partition(s1_grp, cand_grp)
            all_cands.update(cands)
            all_prov.update(prov)

    # 5. Fit Feature Extractor & Build Feature Matrix
    with Timer("Fitting TF-IDF & Building Feature Matrix"):
        extractor = FeatureExtractor()
        extractor.fit_tfidf(s1_norm["name_clean"].tolist() + cand_norm["name_clean"].tolist())

        feature_matrix = build_feature_matrix(
            candidate_pairs=all_cands,
            s1_df=s1_norm,
            cand_df=cand_norm,
            feature_extractor=extractor,
            provenance=all_prov,
            gt_df=gt_val,
            show_progress=True
        )

    # 6. Summary & Health Check
    print("\n--- Feature Matrix Summary ---")
    print(f"Total Candidate Pair Rows: {len(feature_matrix):,}")
    print(f"Feature Columns ({len(feature_matrix.columns) - 3}): {[c for c in feature_matrix.columns if c not in ('s1_id', 'cand_id', 'label')]}")
    
    label_counts = feature_matrix["label"].value_counts().to_dict()
    pos_count = label_counts.get(1.0, 0)
    neg_count = label_counts.get(0.0, 0)
    print(f"Class Balance: Positives = {pos_count:,} ({pos_count / max(1, len(feature_matrix)) * 100:.2f}%) | Negatives = {neg_count:,} ({neg_count / max(1, len(feature_matrix)) * 100:.2f}%)")

    # Check missing values
    null_counts = feature_matrix.isnull().sum().sum()
    print(f"Total Null/Missing Values: {null_counts}")
    assert null_counts == 0, "Feature matrix contains unexpected null values!"

    # Save validation feature matrix
    val_matrix_path = cache_dir / "val_features.parquet"
    feature_matrix.to_parquet(val_matrix_path, index=False, compression="snappy")
    print(f"Saved feature matrix to: {val_matrix_path}")

    print("\n" + "=" * 60)
    print(" Phase 4 Pairwise Feature Engineering Completed Successfully! ")
    print("=" * 60)


if __name__ == "__main__":
    run_feature_pipeline(num_s1_sample=1000)
