"""
Data Profiling & Validation Split Generation Script.
Analyzes dataset shapes, distributions, singleton rates, and creates configs/val_split_seed42.json.
"""

import sys
from pathlib import Path

# Add package root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_io import (
    load_source,
    load_ground_truth,
    create_stratified_validation_split
)
from src.utils import load_config, set_seed, Timer


def run_profiling():
    config = load_config()
    set_seed(config["seed"])
    paths = config["paths"]

    print("=" * 60)
    print(" Amazon ML Challenge 2026 — Dataset Profiling & Split Setup ")
    print("=" * 60)

    # 1. Load Training Data
    with Timer("Loading Source 1 (Reference)"):
        s1 = load_source(paths["train_s1"])
    with Timer("Loading Source 2"):
        s2 = load_source(paths["train_s2"])
    with Timer("Loading Source 3"):
        s3 = load_source(paths["train_s3"])
    with Timer("Loading Ground Truth"):
        gt = load_ground_truth(paths["train_gt"], parse_matches=True)

    print("\n--- Training Data Sizes ---")
    print(f"Source 1 Records: {len(s1):,}")
    print(f"Source 2 Records: {len(s2):,}")
    print(f"Source 3 Records: {len(s3):,}")
    print(f"Ground Truth Records: {len(gt):,}")

    # 2. Country Distributions
    print("\n--- Country Distribution (Train) ---")
    print("Source 1:", s1["country"].value_counts().to_dict())
    print("Source 2:", s2["country"].value_counts().to_dict())
    print("Source 3:", s3["country"].value_counts().to_dict())

    # 3. Ground Truth Matching Statistics
    num_matches = gt["num_matches"]
    singleton_count = (num_matches == 0).sum()
    singleton_pct = singleton_count / len(gt) * 100
    one_match_count = (num_matches == 1).sum()
    multi_match_count = (num_matches > 1).sum()

    print("\n--- Ground Truth Match Distribution ---")
    print(f"Singletons (0 matches): {singleton_count:,} ({singleton_pct:.2f}%)")
    print(f"Single Match (1 match): {one_match_count:,} ({one_match_count / len(gt) * 100:.2f}%)")
    print(f"Multi Matches (>1):     {multi_match_count:,} ({multi_match_count / len(gt) * 100:.2f}%)")
    print(f"Max Matches per S1:     {num_matches.max()}")
    print(f"Mean Matches per S1:    {num_matches.mean():.3f}")

    # 4. Check for Country Crossing in Matches
    print("\n--- Cross-Country Validation Check ---")
    # Build fast country lookup map
    s1_country = dict(zip(s1["entity_id"], s1["country"].str.strip().str.lower()))
    s2_country = dict(zip(s2["entity_id"], s2["country"].str.strip().str.lower()))
    s3_country = dict(zip(s3["entity_id"], s3["country"].str.strip().str.lower()))
    
    country_lookup = {**s2_country, **s3_country}
    cross_country_matches = 0
    total_match_pairs = 0

    for s1_id, matches in zip(gt["source1_entity_id"], gt["match_list"]):
        s1_c = s1_country.get(s1_id)
        for m in matches:
            total_match_pairs += 1
            cand_c = country_lookup.get(m)
            if s1_c and cand_c and s1_c != cand_c:
                cross_country_matches += 1

    print(f"Total True Match Pairs: {total_match_pairs:,}")
    print(f"Cross-Country Match Pairs: {cross_country_matches} ({(cross_country_matches/max(1, total_match_pairs))*100:.4f}%)")
    if cross_country_matches == 0:
        print(">> Confirmed: 100% of matches are strictly within the same country. Country bucketing is 100% recall safe!")

    # 5. Create Stratified Validation Split
    print("\n--- Creating Stratified Validation Split ---")
    split_info = create_stratified_validation_split(
        s1_df=s1,
        gt_df=gt,
        val_ratio=config["validation"]["val_ratio"],
        seed=config["seed"],
        save_path=paths["val_split"]
    )
    stats = split_info["stats"]
    print(f"Split saved to: {paths['val_split']}")
    print(f"Train S1 Entities: {stats['train_count']:,} (Singleton rate: {stats['train_singleton_ratio']*100:.2f}%)")
    print(f"Val S1 Entities:   {stats['val_count']:,} (Singleton rate: {stats['val_singleton_ratio']*100:.2f}%)")
    print("Train Countries:", stats["train_countries"])
    print("Val Countries:  ", stats["val_countries"])

    # 6. Check Test Set Profiles (Without Ground Truth)
    print("\n--- Test Set Overview ---")
    test_s1 = load_source(paths["test_s1"])
    test_s2 = load_source(paths["test_s2"])
    test_s3 = load_source(paths["test_s3"])
    print(f"Test Source 1 Records: {len(test_s1):,}")
    print(f"Test Source 2 Records: {len(test_s2):,}")
    print(f"Test Source 3 Records: {len(test_s3):,}")
    print("Test S1 Countries:", test_s1["country"].value_counts().to_dict())
    print("Test S2 Countries:", test_s2["country"].value_counts().to_dict())
    print("Test S3 Countries:", test_s3["country"].value_counts().to_dict())

    print("\n" + "=" * 60)
    print(" Phase 1 Profiling & Validation Split Completed Successfully! ")
    print("=" * 60)


if __name__ == "__main__":
    run_profiling()
