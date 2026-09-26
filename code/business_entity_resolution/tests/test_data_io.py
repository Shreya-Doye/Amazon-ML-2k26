import io
import json
import pytest
import pandas as pd
from pathlib import Path
from src.data_io import (
    load_source,
    load_ground_truth,
    validate_source_schema,
    validate_ground_truth_schema,
    create_stratified_validation_split,
    load_validation_split
)


def test_validate_source_schema_valid():
    df = pd.DataFrame({
        "entity_id": ["S1-001", "S1-002"],
        "business_name": ["Corp A", "Corp B"],
        "business_address": ["123 Main St", "456 Oak Rd"],
        "country": ["US", "India"]
    })
    validate_source_schema(df, source_hint="source1")


def test_validate_source_schema_missing_col():
    df = pd.DataFrame({
        "entity_id": ["S1-001"],
        "business_name": ["Corp A"],
        "country": ["US"]
    })
    with pytest.raises(ValueError, match="missing columns"):
        validate_source_schema(df)


def test_validate_source_schema_wrong_prefix():
    df = pd.DataFrame({
        "entity_id": ["S2-001"],
        "business_name": ["Corp A"],
        "business_address": ["123 Main St"],
        "country": ["US"]
    })
    with pytest.raises(ValueError, match="Entity ID format violation"):
        validate_source_schema(df, source_hint="source1")


def test_validate_ground_truth_schema_valid():
    df = pd.DataFrame({
        "source1_entity_id": ["S1-001"],
        "matched_entity_ids": ["S2-001,S3-002"]
    })
    validate_ground_truth_schema(df)


def test_stratified_validation_split(tmp_path):
    s1_df = pd.DataFrame({
        "entity_id": [f"S1-{i:03d}" for i in range(100)],
        "business_name": [f"Business {i}" for i in range(100)],
        "business_address": [f"Address {i}" for i in range(100)],
        "country": ["US"] * 50 + ["India"] * 50
    })
    
    # 30% singletons, 70% matched
    gt_df = pd.DataFrame({
        "source1_entity_id": [f"S1-{i:03d}" for i in range(100)],
        "matched_entity_ids": ["" if i % 3 == 0 else f"S2-{i:03d}" for i in range(100)]
    })
    
    split_file = tmp_path / "test_split.json"
    result = create_stratified_validation_split(
        s1_df=s1_df,
        gt_df=gt_df,
        val_ratio=0.20,
        seed=42,
        save_path=split_file
    )
    
    assert len(result["train_s1_ids"]) == 80
    assert len(result["val_s1_ids"]) == 20
    assert len(set(result["train_s1_ids"]).intersection(set(result["val_s1_ids"]))) == 0
    
    # Check reload
    reloaded = load_validation_split(split_file)
    assert reloaded["train_s1_ids"] == result["train_s1_ids"]
    assert reloaded["val_s1_ids"] == result["val_s1_ids"]
