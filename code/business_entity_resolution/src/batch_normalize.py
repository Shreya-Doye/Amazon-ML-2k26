"""
Batch Normalization Pipeline for Entity Resolution.
Learns vocabulary frequency table, initializes Name & Address normalizers,
and normalizes all train/test datasets with parquet caching.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Add package root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

from src.data_io import load_source
from src.utils import load_config, set_seed, Timer
from src.segmentation import WordSegmenter, build_vocab_frequency_dictionary
from src.normalize_name import NameNormalizer
from src.normalize_address import AddressNormalizer


def process_dataframe(
    df: pd.DataFrame,
    name_normalizer: NameNormalizer,
    addr_normalizer: AddressNormalizer,
    batch_size: int = 50000
) -> pd.DataFrame:
    """
    Apply name and address normalization across a DataFrame in chunked batches.
    Returns DataFrame augmented with:
    - name_clean (str)
    - name_tokens (list of str)
    - name_script (str)
    - is_domain_style (bool)
    - name_transliterated (str)
    - addr_clean (str)
    - addr_tokens (list of str)
    - addr_numbers (list of str)
    - addr_landmark_tokens (list of str)
    """
    n = len(df)
    names = df["business_name"].tolist()
    addrs = df["business_address"].tolist()

    name_clean_list = []
    name_tokens_list = []
    name_script_list = []
    is_domain_list = []
    name_trans_list = []

    addr_clean_list = []
    addr_tokens_list = []
    addr_numbers_list = []
    addr_landmarks_list = []

    for name, addr in zip(names, addrs):
        n_rec = name_normalizer.normalize(name)
        a_rec = addr_normalizer.normalize(addr)

        name_clean_list.append(n_rec.clean)
        name_tokens_list.append(n_rec.tokens)
        name_script_list.append(n_rec.script)
        is_domain_list.append(n_rec.is_domain_style)
        name_trans_list.append(n_rec.transliterated)

        addr_clean_list.append(a_rec.clean)
        addr_tokens_list.append(a_rec.tokens)
        addr_numbers_list.append(a_rec.numbers)
        addr_landmarks_list.append(a_rec.landmark_tokens)

    res_df = df.copy()
    res_df["name_clean"] = name_clean_list
    res_df["name_tokens"] = name_tokens_list
    res_df["name_script"] = name_script_list
    res_df["is_domain_style"] = is_domain_list
    res_df["name_transliterated"] = name_trans_list

    res_df["addr_clean"] = addr_clean_list
    res_df["addr_tokens"] = addr_tokens_list
    res_df["addr_numbers"] = addr_numbers_list
    res_df["addr_landmark_tokens"] = addr_landmarks_list

    return res_df


def run_batch_normalization(max_records_per_source: int = None):
    """
    Run full normalization pipeline over train and test sets, caching results to parquet.
    """
    config = load_config()
    set_seed(config["seed"])
    paths = config["paths"]

    cache_dir = Path(paths["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    vocab_freq_path = cache_dir / "vocab_freq.json"

    print("=" * 60)
    print(" Amazon ML Challenge 2026 — Phase 2: Batch Text Normalization ")
    print("=" * 60)

    # 1. Check or build vocabulary frequency dictionary
    segmenter = WordSegmenter()
    if vocab_freq_path.exists():
        print(f"Loading existing vocabulary frequency dictionary from: {vocab_freq_path}")
        segmenter.load(vocab_freq_path)
    else:
        print("Building vocabulary frequency dictionary from training data...")
        with Timer("Building Vocab Frequency Dictionary"):
            s1 = load_source(paths["train_s1"], nrows=max_records_per_source)
            s2 = load_source(paths["train_s2"], nrows=max_records_per_source)
            s3 = load_source(paths["train_s3"], nrows=max_records_per_source)
            all_names = list(s1["business_name"]) + list(s2["business_name"]) + list(s3["business_name"])
            freq_dict = build_vocab_frequency_dictionary(all_names, min_count=2)
            segmenter.load_from_dict(freq_dict)
            segmenter.save(vocab_freq_path)
            print(f"Saved vocabulary frequency dictionary ({len(freq_dict):,} unique words) to {vocab_freq_path}")

    # 2. Instantiate Normalizers
    legal_suffixes_path = paths["legal_suffixes"]
    address_abbrevs_path = paths["address_abbreviations"]
    
    name_normalizer = NameNormalizer(
        legal_suffixes_path=legal_suffixes_path,
        segmenter=segmenter,
        min_segment_len=config["normalization"]["segmentation_min_token_len"]
    )
    addr_normalizer = AddressNormalizer(abbreviations_path=address_abbrevs_path)

    # 3. Spot Check Demonstration
    print("\n--- Spot Check Demonstration ---")
    demo_cases = [
        ("www.maurewilliamscolombier.com", "85 Wayne Rd, Apt 4B, Ticonderoga Twp"),
        ("Maure Williams Pvt. Ltd.", "Shop 12, Near SBI ATM, MG Road, Bangalore"),
        ("भारत एक्सपोर्ट्स लिमिटेड", "Plot 44, Okhla Industrial Area Phase 3, New Delhi"),
        ("Societe Nouvelle SARL", "15 Rue de la Paix, 75002 Paris")
    ]
    for raw_name, raw_addr in demo_cases:
        n_res = name_normalizer.normalize(raw_name)
        a_res = addr_normalizer.normalize(raw_addr)
        print(f"\n[Raw Name]: {raw_name}")
        print(f"  -> Clean:  {n_res.clean}")
        print(f"  -> Tokens: {n_res.tokens} (Script: {n_res.script}, Domain: {n_res.is_domain_style})")
        if n_res.transliterated:
            print(f"  -> Transliterated: {n_res.transliterated}")
        print(f"[Raw Addr]: {raw_addr}")
        print(f"  -> Clean:  {a_res.clean}")
        print(f"  -> Numbers: {a_res.numbers} | Landmarks: {a_res.landmark_tokens}")

    # 4. Sources to normalize and cache
    tasks = [
        ("Train Source 1", paths["train_s1"], cache_dir / "train_s1_normalized.parquet"),
        ("Train Source 2", paths["train_s2"], cache_dir / "train_s2_normalized.parquet"),
        ("Train Source 3", paths["train_s3"], cache_dir / "train_s3_normalized.parquet"),
        ("Test Source 1", paths["test_s1"], cache_dir / "test_s1_normalized.parquet"),
        ("Test Source 2", paths["test_s2"], cache_dir / "test_s2_normalized.parquet"),
        ("Test Source 3", paths["test_s3"], cache_dir / "test_s3_normalized.parquet"),
    ]

    print("\n--- Batch Normalization & Parquet Caching ---")
    for name, src_path, out_path in tasks:
        if out_path.exists():
            print(f"[{name}] Already cached at: {out_path} (Skipping)")
            continue
            
        print(f"\n[{name}] Normalizing from: {src_path}")
        with Timer(f"Normalizing {name}"):
            df = load_source(src_path, nrows=max_records_per_source)
            norm_df = process_dataframe(df, name_normalizer, addr_normalizer)
            norm_df.to_parquet(out_path, index=False, compression="snappy")
            print(f"Successfully cached {len(norm_df):,} records to: {out_path}")

    print("\n" + "=" * 60)
    print(" Phase 2 Text Normalization Completed Successfully! ")
    print("=" * 60)


if __name__ == "__main__":
    run_batch_normalization()
