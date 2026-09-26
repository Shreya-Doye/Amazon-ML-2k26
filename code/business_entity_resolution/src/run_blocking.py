"""
End-to-End Multi-Channel Blocking & Candidate Generation Runner.
Executes country-partitioned multi-channel candidate generation,
computes recall ceiling, and writes candidate_pairs.tsv.
"""

import sys
from pathlib import Path
from typing import Dict, Set

# Add package root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import pandas as pd
from src.data_io import load_source, load_ground_truth, load_validation_split
from src.utils import load_config, set_seed, Timer
from src.blocking.country_bucket import partition_by_country
from src.blocking.union_blocking import MultiChannelBlocker, write_candidate_pairs_tsv
from src.blocking.validate_blocking import evaluate_blocking_recall


def run_blocking_pipeline(
    use_val_split_only: bool = True,
    sample_size: int = 20000,
    write_tsv: bool = True
):
    config = load_config()
    set_seed(config["seed"])
    paths = config["paths"]
    cache_dir = Path(paths["cache_dir"])

    print("=" * 60)
    print(" Amazon ML Challenge 2026 — Phase 3: Multi-Channel Blocking ")
    print("=" * 60)

    # 1. Load normalized parquet or fallback to loading raw
    s1_norm_path = cache_dir / "train_s1_normalized.parquet"
    s2_norm_path = cache_dir / "train_s2_normalized.parquet"
    s3_norm_path = cache_dir / "train_s3_normalized.parquet"

    with Timer("Loading Normalized Data"):
        if s1_norm_path.exists() and s2_norm_path.exists() and s3_norm_path.exists():
            s1 = pd.read_parquet(s1_norm_path)
            s2 = pd.read_parquet(s2_norm_path)
            s3 = pd.read_parquet(s3_norm_path)
        else:
            print("Normalized parquet files not found, running batch normalization...")
            from src.batch_normalize import run_batch_normalization
            run_batch_normalization(max_records_per_source=sample_size)
            s1 = pd.read_parquet(s1_norm_path)
            s2 = pd.read_parquet(s2_norm_path)
            s3 = pd.read_parquet(s3_norm_path)

    gt = load_ground_truth(paths["train_gt"], parse_matches=True)

    if sample_size and len(s1) > sample_size:
        print(f"Subsampling S1 to {sample_size:,} records for fast evaluation...")
        s1 = s1.iloc[:sample_size].copy()

    s1_id_set = set(s1["entity_id"])
    gt_scoped = gt[gt["source1_entity_id"].isin(s1_id_set)].copy()
    
    # 3. Concatenate Source 2 and Source 3 for candidates
    cand_df = pd.concat([s2, s3], ignore_index=True)
    
    # Check what fraction of true matches exist in the candidate DataFrame
    from src.data_io import get_scoped_matched_ids
    true_matched_ids = get_scoped_matched_ids(s1_id_set, gt_scoped)
    available_cand_ids = set(cand_df["entity_id"])
    present_matches = true_matched_ids.intersection(available_cand_ids)
    print(f"\n[Ground Truth Integrity] Required S2/S3 Target IDs: {len(true_matched_ids):,} | Present in S2/S3 pool: {len(present_matches):,} ({(len(present_matches)/max(1, len(true_matched_ids)))*100:.2f}%)")

    # 4. Partition by Country
    s1_by_country = partition_by_country(s1)
    cand_by_country = partition_by_country(cand_df)

    # 5. Initialize MultiChannelBlocker
    blocker = MultiChannelBlocker(
        df_cap_percentile=config["blocking"]["df_cap_percentile"],
        ngram_jaccard_threshold=config["blocking"]["ngram_jaccard_threshold"],
        ann_k=config["blocking"]["ann_k"],
        ann_nlist=config["blocking"]["ann_nlist"]
    )

    all_candidates: Dict[str, Set[str]] = {}
    all_provenance = {}

    print("\n--- Executing Multi-Channel Blocking by Country ---")
    for c_key, s1_group in s1_by_country.items():
        cand_group = cand_by_country.get(c_key, pd.DataFrame(columns=cand_df.columns))
        print(f"\n[Country: {c_key.upper()}] S1: {len(s1_group):,} | Candidates (S2+S3): {len(cand_group):,}")
        
        with Timer(f"Blocking {c_key.upper()}"):
            cands, prov = blocker.block_country_partition(s1_group, cand_group)
            all_candidates.update(cands)
            all_provenance.update(prov)

    # 6. Evaluate Recall Ceiling & Candidate Statistics
    print("\n" + "=" * 60)
    print(" Blocking Evaluation Report (Recall Ceiling) ")
    print("=" * 60)
    stats = evaluate_blocking_recall(all_candidates, gt_scoped)
    
    print(f"Evaluated S1 Entities:     {stats['evaluated_s1_entities']:,}")
    print(f"Total True Match Links:    {stats['total_true_match_links']:,}")
    print(f"Recovered Match Links:     {stats['recovered_match_links']:,}")
    print(f"RECALL CEILING:            {stats['recall_ceiling_pct']:.2f}%")
    print(f"S1 100% Perfect Coverage:  {stats['s1_perfect_coverage_rate']*100:.2f}%")
    print(f"Mean Candidates per S1:    {stats['mean_candidates_per_s1']:.1f}")
    print(f"Median Candidates per S1:  {stats['median_candidates_per_s1']:.0f}")
    print(f"P95 Candidates per S1:     {stats['p95_candidates_per_s1']:.0f}")
    print(f"Max Candidates on one S1:  {stats['max_candidates_per_s1']}")

    # 7. Write candidate_pairs.tsv if requested
    if write_tsv:
        out_cand_path = Path(paths["candidate_pairs_tsv"])
        with Timer("Writing candidate_pairs.tsv"):
            write_candidate_pairs_tsv(all_candidates, s1["entity_id"].tolist(), out_cand_path)
            print(f"Candidate pairs TSV written to: {out_cand_path}")

    print("\n" + "=" * 60)
    print(" Phase 3 Multi-Channel Blocking Completed Successfully! ")
    print("=" * 60)

    return stats


if __name__ == "__main__":
    run_blocking_pipeline(sample_size=5000)
