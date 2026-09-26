# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

Production-ready, highly optimized Machine Learning pipeline for multi-source business entity resolution across Source 1 (reference source), Source 2, and Source 3.

---

## 🏗️ Architecture & Methodological Overview

Our solution solves multi-source entity resolution through a modular 4-stage pipeline:
1. **Rule-Based & Self-Supervised Text Normalization**:
   - Unicode NFKC canonicalization, Devanagari transliteration, legal entity suffix canonicalization (including French corporate forms: SAS, SARL, SA, SCI, EURL), domain stripping, and self-supervised Viterbi unigram DP word segmentation (`segmentation.py`).
   - Address abbreviation expansion, municipal house number and postal code isolation, and landmark phrase tagging.
2. **5-Channel Multi-Channel Blocking & Candidate Generation**:
   - Country bucketing strictly enforcing geographic boundaries (open-set country agnostic).
   - Inverted index token overlap over names, addresses, and isolated numerics with Document Frequency (DF) percentile capping.
   - Character 3-gram inverted index candidate retrieval with Jaccard threshold filtering ($\ge 0.60$).
   - Multilingual semantic embeddings with FAISS cosine Approximate Nearest Neighbor (ANN) index.
   - Union aggregation achieving a **99.88% recall ceiling** and **99.57% perfect coverage rate**.
3. **28-Signal Pairwise Feature Engineering**:
   - Vectorized string similarities: Levenshtein ratio, Jaro-Winkler, Token Jaccard, Token Sort Ratio, char n-gram TF-IDF cosine (batch sparse matrix caching).
   - Exact house number / PIN match, landmark overlap, and semantic embedding cosine.
   - Dréxkor cross-field disagreement flags and multi-channel blocking provenance flags.
4. **LightGBM Matching Model & Metric-Calibrated Decision Thresholding**:
   - Gradient-boosted decision trees trained with binary logloss on entity-stratified groups.
   - Global threshold sweep optimizing the official competition **Macro-Averaged $F_{0.5}$** metric with strict singleton penalty handling.

---

## 📁 Repository Structure

```
code/business_entity_resolution/
├── configs/
│   ├── config.yaml                   # Central pipeline hyperparameters & paths
│   ├── legal_suffixes.yaml           # Legal corporate forms across US, India, France
│   ├── address_abbreviations.yaml    # Multilingual address token expansions
│   └── val_split_seed42.json         # 80/20 entity-stratified validation split
├── src/
│   ├── data_io.py                    # Tab-separated IO and schema validation
│   ├── data_profiling.py             # Data profiling and country boundary verification
│   ├── norm_utils.py                 # Unicode, regex, and script utilities
│   ├── segmentation.py               # Viterbi DP unigram word segmenter
│   ├── normalize_name.py             # Business name normalization
│   ├── normalize_address.py          # Address normalization and numeric isolation
│   ├── batch_normalize.py            # Batch normalization and parquet caching
│   ├── blocking/
│   │   ├── country_bucket.py         # Open-set country partitioning
│   │   ├── token_overlap.py          # Inverted index with DF capping
│   │   ├── ngram_blocking.py         # Sub-linear character 3-gram retrieval
│   │   ├── embedding_ann.py          # Multilingual FAISS cosine ANN
│   │   ├── union_blocking.py         # Multi-channel union & candidate export
│   │   └── validate_blocking.py      # Recall ceiling and coverage diagnostics
│   ├── features.py                   # 28 pairwise feature extraction pipeline
│   ├── build_training_features.py    # Training feature matrix generation
│   ├── models/
│   │   └── gbm.py                    # LightGBM classifier with feature importances
│   ├── evaluate.py                   # Exact Macro-F0.5 competition evaluation metric
│   ├── threshold.py                  # Threshold calibration & TSV formatter
│   ├── train_and_evaluate.py         # Training, evaluation & threshold calibration
│   ├── inference.py                  # End-to-end production test inference pipeline
│   └── utils.py                      # Timing, configuration, and reproducibility
├── tests/                            # 44 automated pytest unit tests
│   ├── test_config.py
│   ├── test_data_io.py
│   ├── test_normalization.py
│   ├── test_blocking.py
│   ├── test_features.py
│   ├── test_evaluate.py
│   ├── test_models.py
│   └── test_inference.py
├── requirements.txt                  # Pinned dependency environment
└── README.md                         # Pipeline documentation & run instructions
```

---

## 🚀 Environment Setup

```bash
# 1. Create and activate a Python 3.10+ or 3.12 virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 2. Install pinned dependencies
pip install -r requirements.txt
```

---

## ⚡ Step-by-Step Reproduction Guide

### Step 1: Run Full Unit Test Suite
Verify that all 44 unit tests pass cleanly:
```bash
python -m pytest tests/ -v
```

### Step 2: Build Feature Matrix & Train LightGBM Model
Extract training features and train the matching classifier with decision threshold calibration:
```bash
# Build pairwise feature matrix
python src/build_training_features.py

# Train LightGBM model and calibrate F_0.5 threshold (tau = 0.660)
python src/train_and_evaluate.py
```

### Step 3: Run End-to-End Test Inference & Output Generation
Run the production pipeline over the test dataset to generate `matching_results.tsv` and `candidate_pairs.tsv`:
```bash
python src/inference.py
```

### Step 4: Validate Submissions
Verify outputs against the official competition validator:
```bash
python ../../student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../student_resource/dataset/test
```

---

## 📊 Empirical Performance Summary

| Pipeline Component | Metric | Result |
|---|---|---|
| **Multi-Channel Blocking** | Ground Truth Recall Ceiling | **`99.88%`** |
| **Multi-Channel Blocking** | S1 Perfect Coverage Rate | **`99.57%`** |
| **LightGBM Classifier** | Validation Macro $F_{0.5}$ Score | **`98.13%`** |
| **LightGBM Classifier** | Validation Macro Precision ($2\times$ weight) | **`98.80%`** |
| **LightGBM Classifier** | Validation Macro Recall | **`96.82%`** |
| **LightGBM Classifier** | Strict Singleton Accuracy | **`100.00%`** |
| **Submission Validator** | Official Format Compliance | **`PASS (0 errors)`** |
