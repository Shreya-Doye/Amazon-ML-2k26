"""
End-to-End Test Inference Pipeline for Amazon ML Challenge 2026.
Executes the full production workflow:
1. Loads test datasets (including open-set France records).
2. Normalizes entity names and addresses.
3. Performs multi-channel blocking (token overlap, char 3-gram, semantic ANN).
4. Exports candidate_pairs.tsv.
5. Extracts 28 pairwise features.
6. Scores candidate pairs using trained LightGBM model.
7. Applies calibrated decision threshold (tau=0.660).
8. Exports matching_results.tsv.
9. Validates submissions against official validator.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

# Add package root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import pandas as pd
import numpy as np

from src.data_io import load_source
from src.utils import load_config, set_seed, Timer
from src.normalize_name import NameNormalizer
from src.normalize_address import AddressNormalizer
from src.segmentation import WordSegmenter
from src.batch_normalize import process_dataframe
from src.blocking.country_bucket import partition_by_country
from src.blocking.union_blocking import MultiChannelBlocker, write_candidate_pairs_tsv
from src.features import FeatureExtractor, build_feature_matrix
from src.models.gbm import MatchingGBM
from src.threshold import format_predictions_by_threshold, write_matching_results_tsv


def run_inference(
    test_s1_path: Optional[str] = None,
    test_s2_path: Optional[str] = None,
    test_s3_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    threshold: float = 0.660,
    sample_s1_limit: Optional[int] = None,
    run_validator: bool = True
):
    config = load_config()
    set_seed(config["seed"])
    paths = config["paths"]

    s1_file = Path(test_s1_path or paths["test_s1"])
    s2_file = Path(test_s2_path or paths["test_s2"])
    s3_file = Path(test_s3_path or paths["test_s3"])
    out_dir = Path(output_dir or paths["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(paths["cache_dir"])

    cand_out_file = out_dir / "candidate_pairs.tsv"
    match_out_file = out_dir / "matching_results.tsv"

    print("=" * 70)
    print(" Amazon ML Challenge 2026 — End-to-End Test Inference Pipeline ")
    print("=" * 70)

    # 1. Load Test Sources
    with Timer("Loading Test Sources"):
        s1_df = load_source(s1_file)
        s2_df = load_source(s2_file)
        s3_df = load_source(s3_file)

        if sample_s1_limit is not None and sample_s1_limit < len(s1_df):
            print(f"[Sampling Active] Limiting S1 entities to first {sample_s1_limit:,}")
            s1_df = s1_df.head(sample_s1_limit).copy()

        all_test_s1_ids = s1_df["entity_id"].tolist()
        cand_df_raw = pd.concat([s2_df, s3_df], ignore_index=True)

        print(f"Loaded {len(s1_df):,} Source 1 test records.")
        print(f"Loaded {len(s2_df):,} Source 2 and {len(s3_df):,} Source 3 test records (Total pool: {len(cand_df_raw):,}).")
        countries_present = s1_df["country"].str.upper().unique().tolist()
        print(f"Countries present in test set: {countries_present}")

    # 2. Text Normalization
    with Timer("Normalizing Test Records"):
        segmenter = WordSegmenter()
        vocab_path = cache_dir / "vocab_freq.json"
        if vocab_path.exists():
            segmenter.load(vocab_path)

        name_norm = NameNormalizer(paths["legal_suffixes"], segmenter=segmenter)
        addr_norm = AddressNormalizer(paths["address_abbreviations"])

        s1_norm = process_dataframe(s1_df, name_norm, addr_norm)
        cand_norm = process_dataframe(cand_df_raw, name_norm, addr_norm)

    # 3. Multi-Channel Blocking
    with Timer("Executing Multi-Channel Blocking across Country Partitions"):
        s1_by_c = partition_by_country(s1_norm)
        cand_by_c = partition_by_country(cand_norm)

        blocker = MultiChannelBlocker(
            df_cap_percentile=config["blocking"]["df_cap_percentile"],
            ngram_jaccard_threshold=config["blocking"]["ngram_jaccard_threshold"]
        )

        all_cands: Dict[str, Set[str]] = {}
        all_prov: Dict[Tuple[str, str], Dict[str, bool]] = {}

        for c_key, s1_grp in s1_by_c.items():
            cand_grp = cand_by_c.get(c_key, pd.DataFrame(columns=cand_norm.columns))
            print(f"  -> Blocking partition '{c_key}': {len(s1_grp):,} S1 vs {len(cand_grp):,} Candidates...")
            cands, prov = blocker.block_country_partition(s1_grp, cand_grp)
            all_cands.update(cands)
            all_prov.update(prov)

        # Export Candidate Pairs TSV
        write_candidate_pairs_tsv(all_cands, all_test_s1_ids, cand_out_file)
        total_pairs = sum(len(c) for c in all_cands.values())
        print(f"Generated {total_pairs:,} candidate pairs across {len(all_test_s1_ids):,} S1 entities.")
        print(f"Saved candidate pairs to: {cand_out_file}")

    # 4. Feature Extraction
    with Timer("Fitting TF-IDF & Extracting Pairwise Features"):
        extractor = FeatureExtractor()
        extractor.fit_tfidf(s1_norm["name_clean"].tolist() + cand_norm["name_clean"].tolist())

        feature_matrix = build_feature_matrix(
            candidate_pairs=all_cands,
            s1_df=s1_norm,
            cand_df=cand_norm,
            feature_extractor=extractor,
            provenance=all_prov,
            gt_df=None,  # No ground truth for test
            show_progress=True
        )
        print(f"Extracted feature matrix shape: {feature_matrix.shape}")

    # 5. Matching Model Inference
    with Timer("Predicting Match Probabilities with LightGBM"):
        model_path = cache_dir / "lgbm_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}! Run train_and_evaluate.py first.")

        model = MatchingGBM.load(model_path)
        feature_cols = [c for c in feature_matrix.columns if c not in ("s1_id", "cand_id", "label")]
        X_test = feature_matrix[feature_cols]

        if len(X_test) > 0:
            test_probs = model.predict_proba(X_test)
        else:
            test_probs = np.array([])

    # 6. Apply Decision Threshold & Generate matching_results.tsv
    with Timer("Formatting Predictions & Writing matching_results.tsv"):
        if len(feature_matrix) > 0:
            predictions = format_predictions_by_threshold(
                s1_series=feature_matrix["s1_id"],
                cand_series=feature_matrix["cand_id"],
                probs=test_probs,
                threshold=threshold,
                all_s1_ids=all_test_s1_ids
            )
        else:
            predictions = {s1_id: [] for s1_id in all_test_s1_ids}

        write_matching_results_tsv(
            predictions=predictions,
            output_path=match_out_file,
            all_s1_ids=all_test_s1_ids
        )

        num_matched_s1 = sum(1 for v in predictions.values() if len(v) > 0)
        num_singletons = len(all_test_s1_ids) - num_matched_s1
        total_links = sum(len(v) for v in predictions.values())
        print(f"Matched S1 Entities: {num_matched_s1:,} | Singletons: {num_singletons:,} | Total Links Predicted: {total_links:,}")
        print(f"Saved matching results to: {match_out_file}")

    # 7. Official Submission Validation
    if run_validator:
        print("\n" + "-" * 70)
        print(" Running Official Challenge Submission Validator... ")
        print("-" * 70)
        validator_script = Path("student_resource/utils/validate_submission.py")
        if validator_script.exists():
            test_dir = s1_file.parent
            cmd = [
                sys.executable,
                str(validator_script),
                "--matching", str(match_out_file),
                "--candidate", str(cand_out_file),
                "--test-dir", str(test_dir)
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            print(res.stdout)
            if res.stderr:
                print(res.stderr)
            if res.returncode == 0:
                print(">>> STATUS: PASS — Official validator accepted the submission package! <<<")
            else:
                print(">>> STATUS: FAIL — Validator reported issues. Check above log. <<<")
        else:
            print(f"Validator script not found at {validator_script}, skipping.")

    print("\n" + "=" * 70)
    print(" Inference Pipeline Completed Successfully! ")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Inference for Business Entity Resolution")
    parser.add_argument("--test-s1", type=str, default=None, help="Path to test_source1.tsv")
    parser.add_argument("--test-s2", type=str, default=None, help="Path to test_source2.tsv")
    parser.add_argument("--test-s3", type=str, default=None, help="Path to test_source3.tsv")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for output TSVs")
    parser.add_argument("--threshold", type=float, default=0.660, help="Decision threshold tau")
    parser.add_argument("--sample-s1", type=int, default=None, help="Sample limit on S1 for testing")
    parser.add_argument("--no-validate", action="store_true", help="Skip validator script")

    args = parser.parse_args()
    run_inference(
        test_s1_path=args.test_s1,
        test_s2_path=args.test_s2,
        test_s3_path=args.test_s3,
        output_dir=args.output_dir,
        threshold=args.threshold,
        sample_s1_limit=args.sample_s1,
        run_validator=not args.no_validate
    )
