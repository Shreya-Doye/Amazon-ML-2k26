import pytest
import numpy as np
import pandas as pd
from src.features import (
    jaccard_similarity,
    extract_suffix_token,
    FeatureExtractor,
    build_feature_matrix
)


def test_jaccard_similarity():
    assert jaccard_similarity(["a", "b"], ["a", "b"]) == 1.0
    assert jaccard_similarity(["a", "b"], ["b", "c"]) == 1.0 / 3.0
    assert jaccard_similarity([], []) == 1.0
    assert jaccard_similarity(["a"], []) == 0.0


def test_extract_suffix_token():
    assert extract_suffix_token(["maure", "williams", "PVT_LTD_TOKEN"]) == "PVT_LTD_TOKEN"
    assert extract_suffix_token(["tech", "solutions", "CORP_TOKEN"]) == "CORP_TOKEN"
    assert extract_suffix_token(["regular", "name"]) is None


def test_pairwise_feature_extractor_drexkor_case():
    extractor = FeatureExtractor()
    
    # Dréxkor case: trade name differs, but address and number match perfectly
    s1_row = {
        "entity_id": "S1-00001",
        "name_clean": "drexkor enterprise",
        "name_tokens": ["drexkor", "enterprise"],
        "addr_clean": "85 wayne avenue ticonderoga TOWNSHIP_TOKEN",
        "addr_tokens": ["wayne", "avenue", "ticonderoga", "TOWNSHIP_TOKEN"],
        "addr_numbers": ["85"],
        "addr_landmark_tokens": []
    }
    cand_row = {
        "entity_id": "S2-00047",
        "name_clean": "ticonderoga venture",
        "name_tokens": ["ticonderoga", "venture"],
        "addr_clean": "85 wanye ave ticonderoga twp",
        "addr_tokens": ["wanye", "avenue", "ticonderoga", "TOWNSHIP_TOKEN"],
        "addr_numbers": ["85"],
        "addr_landmark_tokens": []
    }
    
    prov = {"ch_tok_name": False, "ch_tok_addr": True, "ch_ngram": False, "ch_ann_name": False, "ch_ann_addr": False}
    feats = extractor.compute_pair_features(s1_row, cand_row, provenance=prov)

    # Assert Dréxkor cross-field interaction is activated
    assert feats["name_levenshtein_ratio"] < 0.45
    assert feats["addr_token_jaccard"] >= 0.60
    assert feats["name_low_addr_high"] == 1.0
    assert feats["name_high_addr_low"] == 0.0
    
    # Assert numeric exact match
    assert feats["addr_numeric_exact_match"] == 1.0
    assert feats["addr_numeric_missing_flag"] == 0.0

    # Assert provenance
    assert feats["is_source2"] == 1.0
    assert feats["prov_tok_addr"] == 1.0
    assert feats["prov_tok_name"] == 0.0


def test_build_feature_matrix():
    extractor = FeatureExtractor()
    extractor.fit_tfidf(["maure williams", "tech solutions", "global trade"])

    s1_df = pd.DataFrame([{
        "entity_id": "S1-001",
        "name_clean": "maure williams",
        "name_tokens": ["maure", "williams"],
        "addr_clean": "123 main street",
        "addr_tokens": ["main", "street"],
        "addr_numbers": ["123"],
        "addr_landmark_tokens": []
    }])

    cand_df = pd.DataFrame([
        {
            "entity_id": "S2-001",
            "name_clean": "maure williams",
            "name_tokens": ["maure", "williams"],
            "addr_clean": "123 main street",
            "addr_tokens": ["main", "street"],
            "addr_numbers": ["123"],
            "addr_landmark_tokens": []
        },
        {
            "entity_id": "S3-002",
            "name_clean": "completely different",
            "name_tokens": ["completely", "different"],
            "addr_clean": "456 oak road",
            "addr_tokens": ["oak", "road"],
            "addr_numbers": ["456"],
            "addr_landmark_tokens": []
        }
    ])

    cands = {"S1-001": {"S2-001", "S3-002"}}
    gt_df = pd.DataFrame({
        "source1_entity_id": ["S1-001"],
        "matched_entity_ids": ["S2-001"]
    })

    mat = build_feature_matrix(cands, s1_df, cand_df, extractor, gt_df=gt_df, show_progress=False)

    assert len(mat) == 2
    assert "label" in mat.columns
    
    row_pos = mat[mat["cand_id"] == "S2-001"].iloc[0]
    row_neg = mat[mat["cand_id"] == "S3-002"].iloc[0]

    assert row_pos["label"] == 1.0
    assert row_pos["name_exact_match"] == 1.0
    assert row_pos["addr_exact_match"] == 1.0

    assert row_neg["label"] == 0.0
    assert row_neg["name_exact_match"] == 0.0
