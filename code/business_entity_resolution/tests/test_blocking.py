import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.blocking.country_bucket import country_key, partition_by_country
from src.blocking.token_overlap import InvertedTokenIndex
from src.blocking.ngram_blocking import InvertedNgramIndex, extract_char_ngrams
from src.blocking.embedding_ann import FAISSANNIndex
from src.blocking.union_blocking import MultiChannelBlocker, write_candidate_pairs_tsv
from src.blocking.validate_blocking import evaluate_blocking_recall


def test_country_key_normalization():
    assert country_key(" US ") == "us"
    assert country_key("India") == "india"
    assert country_key("France") == "france"
    assert country_key(None) == "unknown"


def test_partition_by_country():
    df = pd.DataFrame({
        "entity_id": ["1", "2", "3"],
        "country": ["US", "India", "us"]
    })
    parts = partition_by_country(df)
    assert "us" in parts
    assert "india" in parts
    assert len(parts["us"]) == 2
    assert len(parts["india"]) == 1


def test_inverted_token_index_df_capping():
    # Record 1..100 have common token 'common', but Record 1 has 'rare_token'
    records = [(f"S2-{i:03d}", ["common", f"unique_{i}"]) for i in range(100)]
    records.append(("S2-999", ["common", "rare_token"]))

    idx = InvertedTokenIndex(df_cap_percentile=0.90, max_df=50).build(records)
    
    # Query with common token only -> should be capped / return empty
    cands_common = idx.query(["common"])
    assert len(cands_common) == 0

    # Query with rare token -> should return S2-999
    cands_rare = idx.query(["rare_token"])
    assert "S2-999" in cands_rare


def test_inverted_ngram_index():
    records = [
        ("S2-001", "maure williams"),
        ("S2-002", "johnson tech"),
        ("S2-003", "maure willams")  # Missing single letter typo in williams
    ]
    idx = InvertedNgramIndex(n=3, min_jaccard=0.60).build(records)

    cands = idx.query("maure williams", min_jaccard=0.60)
    assert "S2-001" in cands
    assert "S2-003" in cands
    assert "S2-002" not in cands


def test_faiss_ann_index():
    # 3 vectors in 4 dimensions
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0]
    ], dtype=np.float32)
    # L2 normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    ann = FAISSANNIndex(dimension=4).build(embeddings, ["E-1", "E-2", "E-3"])
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    res = ann.search(query, k=2)

    assert "E-1" in res[0]
    assert "E-3" in res[0]
    assert "E-2" not in res[0]


def test_multi_channel_blocker_and_provenance():
    s1_df = pd.DataFrame({
        "entity_id": ["S1-001"],
        "name_clean": ["maure williams"],
        "name_tokens": [["maure", "williams"]],
        "addr_clean": ["85 wayne road"],
        "addr_tokens": [["wayne", "road"]],
        "addr_numbers": [["85"]]
    })
    cand_df = pd.DataFrame({
        "entity_id": ["S2-001", "S3-002"],
        "name_clean": ["maure williams", "unrelated corp"],
        "name_tokens": [["maure", "williams"], ["unrelated", "corp"]],
        "addr_clean": ["85 wayne road", "85 wayne road"],
        "addr_tokens": [["wayne", "road"], ["wayne", "road"]],
        "addr_numbers": [["85"], ["85"]]
    })

    blocker = MultiChannelBlocker(df_cap_percentile=1.0, ngram_jaccard_threshold=0.60)
    cands, prov = blocker.block_country_partition(s1_df, cand_df)

    assert "S2-001" in cands["S1-001"]
    assert "S3-002" in cands["S1-001"]
    
    # S2-001 matches both name and addr
    assert prov[("S1-001", "S2-001")]["ch_tok_name"] is True
    assert prov[("S1-001", "S2-001")]["ch_tok_addr"] is True
    
    # S3-002 matches addr only
    assert prov[("S1-001", "S3-002")]["ch_tok_name"] is False
    assert prov[("S1-001", "S3-002")]["ch_tok_addr"] is True


def test_write_candidate_pairs_and_evaluation(tmp_path):
    cands = {
        "S1-001": {"S2-001", "S3-001"},
        "S1-002": set()  # Singleton
    }
    all_s1 = ["S1-001", "S1-002"]
    out_file = tmp_path / "candidate_pairs.tsv"
    
    write_candidate_pairs_tsv(cands, all_s1, out_file)
    assert out_file.exists()

    gt_df = pd.DataFrame({
        "source1_entity_id": ["S1-001", "S1-002"],
        "matched_entity_ids": ["S2-001,S3-001", ""]
    })

    stats = evaluate_blocking_recall(cands, gt_df)
    assert stats["recall_ceiling"] == 1.0
    assert stats["s1_perfect_coverage_rate"] == 1.0
