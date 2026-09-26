# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

## Overview
This repository contains the end-to-end Machine Learning solution for the Amazon ML Challenge 2026 Business Entity Resolution problem.
The pipeline links multi-source records (Source 2 and Source 3) to canonical reference records (Source 1) with optimized Macro-$F_{0.5}$ score.

## Architecture
1. **Data I/O & Validation**: Tab-separated loading with strict string types and entity-level stratified validation split.
2. **Text Normalization**: Script detection, Unicode NFKC normalization, legal suffix mapping, address abbreviation canonicalization, and custom dataset-adaptive Viterbi word segmentation.
3. **Multi-Channel Blocking**:
   - Inverted index on business name tokens (with Document Frequency capping)
   - Inverted index on address tokens & isolated numeric components
   - Character 3-gram index with Jaccard filtering
   - Multilingual dense semantic embeddings (`Qwen/Qwen3-Embedding-0.6B` / `BAAI/bge-m3`) with FAISS `IndexIVFPQ`
4. **Pairwise Feature Engineering**: Levenshtein ratio, Jaro-Winkler, Jaccard token overlap, character TF-IDF cosine, semantic cosine, numeric address match flags, cross-field disagreement flags, and blocking provenance.
5. **Matching Ensemble**: Sequence-pair classification cross-encoder (`microsoft/mdeberta-v3-base`) ensembled with Gradient Boosted Decision Trees (`LightGBM`).
6. **Threshold Calibration**: Precision-favored thresholding calibrated explicitly for Macro-$F_{0.5}$.

## Quick Start & Reproduction

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest tests/
```
