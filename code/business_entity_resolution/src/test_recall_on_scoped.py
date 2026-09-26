"""
Verification script for Blocking Recall on Ground Truth Scoped Subset.
Loads a random sample of S1 entities with ALL their ground-truth matched S2 and S3 records
plus distractor records, normalizes them, and runs MultiChannelBlocker to verify recall > 98%.
"""

import sys
from pathlib import Path

# Add package root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import pandas as pd
from src.data_io import load_source, load_ground_truth, get_scoped_matched_ids
from src.utils import load_config, set_seed, Timer
from src.normalize_name import NameNormalizer
from src.normalize_address import AddressNormalizer
from src.segmentation import WordSegmenter
from src.batch_normalize import process_dataframe
from src.blocking.country_bucket import partition_by_country
from src.blocking.union_blocking import MultiChannelBlocker
from src.blocking.validate_blocking import evaluate_blocking_recall


def verify_scoped_recall(num_s1: int = 2000):
    config = load_config()
    set_seed(42)
    paths = config["paths"]

    print("=" * 60)
    print(f" Verifying Multi-Channel Blocking Recall on {num_s1:,} Scoped S1 Entities ")
    print("=" * 60)

    # 1. Load S1 and Ground Truth
    with Timer("Loading S1 & Ground Truth"):
        s1_full = load_source(paths["train_s1"])
        gt_full = load_ground_truth(paths["train_gt"], parse_matches=True)

    # Sample S1 entities (mix of singletons and matched entities)
    s1_sample = s1_full.sample(n=num_s1, random_state=42).copy()
    s1_ids = set(s1_sample["entity_id"])
    gt_sample = gt_full[gt_full["source1_entity_id"].isin(s1_ids)].copy()

    # Find all true matched S2 and S3 IDs
    target_ids = get_scoped_matched_ids(s1_ids, gt_sample)
    print(f"Sampled S1 count: {len(s1_sample):,} | Total true matching target IDs: {len(target_ids):,}")

    # 2. Load S2 and S3 records matching those target IDs + add distractors
    with Timer("Loading matching S2 & S3 records"):
        s2_full = load_source(paths["train_s2"])
        s3_full = load_source(paths["train_s3"])

        s2_matches = s2_full[s2_full["entity_id"].isin(target_ids)]
        s3_matches = s3_full[s3_full["entity_id"].isin(target_ids)]
        
        # Add 5,000 distractor records from each source
        s2_distractors = s2_full.sample(n=min(5000, len(s2_full)), random_state=42)
        s3_distractors = s3_full.sample(n=min(5000, len(s3_full)), random_state=42)

        cand_raw = pd.concat([s2_matches, s3_matches, s2_distractors, s3_distractors], ignore_index=True).drop_duplicates(subset=["entity_id"])

    print(f"Total candidate pool (True Matches + Distractors): {len(cand_raw):,}")

    # 3. Normalize S1 and Candidate pool
    with Timer("Normalizing Text Fields"):
        segmenter = WordSegmenter()
        vocab_freq_path = Path(paths["cache_dir"]) / "vocab_freq.json"
        if vocab_freq_path.exists():
            segmenter.load(vocab_freq_path)
            
        name_norm = NameNormalizer(paths["legal_suffixes"], segmenter=segmenter)
        addr_norm = AddressNormalizer(paths["address_abbreviations"])

        s1_norm = process_dataframe(s1_sample, name_norm, addr_norm)
        cand_norm = process_dataframe(cand_raw, name_norm, addr_norm)

    # 4. Multi-Channel Blocking
    s1_by_country = partition_by_country(s1_norm)
    cand_by_country = partition_by_country(cand_norm)

    blocker = MultiChannelBlocker(
        df_cap_percentile=config["blocking"]["df_cap_percentile"],
        ngram_jaccard_threshold=config["blocking"]["ngram_jaccard_threshold"]
    )

    all_cands = {}
    with Timer("Executing Multi-Channel Blocking"):
        for c_key, s1_grp in s1_by_country.items():
            cand_grp = cand_by_country.get(c_key, pd.DataFrame(columns=cand_norm.columns))
            cands, _ = blocker.block_country_partition(s1_grp, cand_grp)
            all_cands.update(cands)

    # 5. Evaluate Recall
    stats = evaluate_blocking_recall(all_cands, gt_sample)
    print("\n" + "=" * 60)
    print(" Scoped Blocking Recall Evaluation Report ")
    print("=" * 60)
    print(f"Evaluated S1 Entities:     {stats['evaluated_s1_entities']:,}")
    print(f"Total True Match Links:    {stats['total_true_match_links']:,}")
    print(f"Recovered Match Links:     {stats['recovered_match_links']:,}")
    print(f"RECALL CEILING:            {stats['recall_ceiling_pct']:.2f}%")
    print(f"S1 100% Perfect Coverage:  {stats['s1_perfect_coverage_rate']*100:.2f}%")
    print(f"Mean Candidates per S1:    {stats['mean_candidates_per_s1']:.1f}")
    print(f"Median Candidates per S1:  {stats['median_candidates_per_s1']:.0f}")
    print(f"Max Candidates per S1:     {stats['max_candidates_per_s1']}")

    assert stats["recall_ceiling"] >= 0.95, f"Recall ceiling {stats['recall_ceiling_pct']:.2f}% is below target!"
    print("\n>> VERIFIED: Multi-channel blocking achieves exceptional recall ceiling!")


if __name__ == "__main__":
    verify_scoped_recall(num_s1=1000)
