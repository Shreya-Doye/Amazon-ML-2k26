import pytest
import pandas as pd
from pathlib import Path
from src.inference import run_inference


def test_inference_end_to_end_dummy(tmp_path):
    # 1. Create dummy test files (including US, India, and open-set France)
    s1_data = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-001\tAcme Tech Corp\t123 Main St Ste 400, New York, NY\tUS\n"
        "S1-002\tBoulangerie Patisserie Paul SAS\t15 Rue de Paris, 75001 Paris\tFrance\n"
        "S1-003\tSharma Enterprises Pvt Ltd\tPlot 45, MIDC Industrial Area, Andheri East, Mumbai\tIndia\n"
        "S1-004\tUnique Singleton Without Match LLC\t999 Desert Rd, Phoenix, AZ\tUS\n"
    )
    s2_data = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-001\tAcme Tech\t123 Main Street #400, NY\tUS\n"
        "S2-002\tPaul Boulangerie\t15 Rue de Paris, Paris\tFrance\n"
    )
    s3_data = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S3-001\tSharma Enterprises\tPlot 45 MIDC Andheri East Mumbai\tIndia\n"
        "S3-002\tDistractor Random LLC\t100 Random Blvd, Dallas, TX\tUS\n"
    )

    s1_path = tmp_path / "test_source1.tsv"
    s2_path = tmp_path / "test_source2.tsv"
    s3_path = tmp_path / "test_source3.tsv"

    s1_path.write_text(s1_data, encoding="utf-8")
    s2_path.write_text(s2_data, encoding="utf-8")
    s3_path.write_text(s3_data, encoding="utf-8")

    out_dir = tmp_path / "output"

    # Run inference without invoking global validator (since dummy files are in tmp_path)
    run_inference(
        test_s1_path=str(s1_path),
        test_s2_path=str(s2_path),
        test_s3_path=str(s3_path),
        output_dir=str(out_dir),
        threshold=0.50,
        run_validator=False
    )

    cand_file = out_dir / "candidate_pairs.tsv"
    match_file = out_dir / "matching_results.tsv"

    assert cand_file.exists(), "candidate_pairs.tsv was not generated!"
    assert match_file.exists(), "matching_results.tsv was not generated!"

    cand_df = pd.read_csv(cand_file, sep="\t", dtype=str, keep_default_na=False)
    match_df = pd.read_csv(match_file, sep="\t", dtype=str, keep_default_na=False)

    assert list(cand_df.columns) == ["source1_entity_id", "candidate_entity_ids"]
    assert list(match_df.columns) == ["source1_entity_id", "matched_entity_ids"]

    # All 4 S1 entities must be present in both files
    assert set(cand_df["source1_entity_id"]) == {"S1-001", "S1-002", "S1-003", "S1-004"}
    assert set(match_df["source1_entity_id"]) == {"S1-001", "S1-002", "S1-003", "S1-004"}

    # S1-004 is a singleton with no matches
    s1_004_matches = match_df[match_df["source1_entity_id"] == "S1-004"]["matched_entity_ids"].iloc[0]
    assert s1_004_matches == ""
