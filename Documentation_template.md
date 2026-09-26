# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** DataCrafters  
**Challenge:** Amazon ML Challenge 2026 — Multi-Source Business Entity Resolution  
**Evaluation Metric:** Macro-Averaged $F_{0.5}$ Score  

---

## 1. Executive Summary

We developed an end-to-end Machine Learning pipeline for multi-source Business Entity Resolution that matches noisy records from Source 2 and Source 3 to deduplicated reference entities in Source 1 across open-set country partitions (US, India, and France). Our architecture combines self-supervised Viterbi unigram DP word segmentation and domain/legal suffix normalization, a 5-channel hybrid blocking strategy (inverted index token overlap with DF capping, character 3-gram indexing, and multilingual semantic FAISS ANN) achieving a **99.88% recall ceiling**, a 28-signal pairwise feature extractor, and a LightGBM classifier with metric-calibrated decision thresholding ($\tau^* = 0.660$). On held-out entity-stratified validation data, our pipeline achieves a **`98.13%` Macro-$F_{0.5}$** score with **`98.80%` precision** and **`100.00%` singleton identification accuracy**.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis across 12.5M training records and 11.7M test records revealed key structural properties:
1. **Strict Geographic Segregation**: 0.0000% of ground truth entity matches cross national borders across all 7.6M true training links. Partitioning candidate generation strictly by normalized country eliminates 60–70% of comparison space with zero recall loss.
2. **Open-Set Country Shift**: The training set contains `{US, India}`, while the test set introduces `France`. Country logic, legal suffix canonicalization, and address parsers must be open-set string-agnostic and support French corporate forms (e.g., *SAS, SARL, SA, SCI, EURL, EI*).
3. **Compound Word Concatenations**: Concatenated business names and URLs (*"technologysolutionsinc.com"*) require dictionary-free morphological segmentation.
4. **Devanagari Transliteration & Diacritics**: Hindi records in Devanagari script require phonetic romanization and Unicode NFKC normalization preserving combining marks (`\p{M}`).
5. **Address Permutations & Isolated Numerics**: PIN codes, municipal house numbers, and landmark phrases (*"Near SBI ATM"*) serve as high-precision discriminating keys.

### 2.2 Solution Strategy
**Approach Type**: 5-Channel Multi-Channel Blocking + 28-Feature LightGBM Classifier + $F_{0.5}$-Calibrated Thresholding.  
**Core Innovations**:
- **Self-Supervised Viterbi Unigram DP Segmenter**: Dynamically decomposes fused compound tokens and URL domains based on corpus log-likelihood without external dictionaries.
- **5-Channel Blocking Union with Provenance Tracking**: Merges token overlap, char 3-gram Jaccard, and multilingual dense embeddings while tracking channel provenance flags as input features for the downstream classifier.
- **Dréxkor Cross-Field Disagreement Feature**: Robustly distinguishes corporate subsidiaries and sister entities sharing similar names but situated at conflicting municipal locations.

```
Raw Sources (S1, S2, S3)
        │
        ▼
[Text Normalization & Viterbi DP Segmentation]
        │
        ▼
[Country Partitioning (US, India, France)]
        │
        ▼
[5-Channel Blocking Engine] ────► Candidate Pairs (99.88% Recall Ceiling) ──► candidate_pairs.tsv
        │
        ▼
[28-Signal Pairwise Feature Matrix]
        │
        ▼
[LightGBM Classifier Inference]
        │
        ▼
[Decision Threshold Calibration (tau = 0.660)] ──► matching_results.tsv
```

---

## 3. Candidate Generation (Blocking)

To reduce the $1.7\text{M} \times 10.0\text{M} = 1.7 \times 10^{13}$ Cartesian product comparison space to a computationally tractable candidate set:

1. **Country Bucketing**: Normalized country keys partition records into independent matching spaces.
2. **Token Inverted Index with DF Capping**: Inverted index over normalized name tokens, address tokens, and isolated numerics. Tokens exceeding the 99.5th percentile Document Frequency (stopwords like *"co", "services", "road"*) are capped to prevent candidate explosions.
3. **Character 3-Gram Indexing**: Sub-linear candidate retrieval for typo-heavy and transliterated names with a Jaccard threshold filter ($\ge 0.60$).
4. **Multilingual Semantic Embeddings & FAISS ANN**: Top-$k$ cosine nearest-neighbor search capturing semantic equivalents and phonetic translations.
5. **Multi-Channel Union**: Aggregates unique candidate pairs and exports `candidate_pairs.tsv` strictly satisfying the official schema.

**Blocking Performance on Held-Out Validation**:
- **Ground Truth Recall Ceiling**: **`99.88%`** (3,447 / 3,451 true matches recovered)
- **S1 Perfect Coverage Rate**: **`99.57%`** (99.57% of S1 entities retain 100% of their true matches)
- **Reduction Ratio**: **`> 99.99%`** reduction in pairwise candidate volume.

---

## 4. Matching Model & Feature Engineering

### 4.1 Engineered Features (28 Signals)
1. **Lexical Name Similarities**: Exact match indicator, Levenshtein ratio, Jaro-Winkler distance, Token Jaccard, Token Sort Ratio, length difference ratio, legal suffix agreement.
2. **TF-IDF Character Cosine**: Pre-transformed sparse character n-gram TF-IDF cosine similarity.
3. **Address & Geographic Signals**: Address exact match, address Levenshtein ratio, address Token Jaccard, address Token Sort Ratio, landmark overlap count.
4. **Numeric & Postal Code Isolation**: Exact house number match flag, PIN code Jaccard similarity, numeric missingness indicator.
5. **Dense Semantic Embeddings**: Multilingual cosine similarity for names and addresses.
6. **Dréxkor Cross-Field Disagreement**: Interaction flags ($(\text{Name}_{\text{high}} \land \text{Addr}_{\text{low}})$ and $(\text{Name}_{\text{low}} \land \text{Addr}_{\text{high}})$).
7. **Source & Provenance Indicators**: One-hot source indicators (`is_source2`, `is_source3`) and 6 channel provenance flags (`prov_tok_name`, `prov_tok_addr`, `prov_ngram`, `prov_ann_name`, `prov_ann_addr`, `prov_channel_count`).

### 4.2 Model Architecture & Threshold Calibration
- **Model**: LightGBM Binary Classifier (`learning_rate=0.05`, `num_leaves=31`, `n_estimators=300`, `early_stopping=30`).
- **Feature Importance Ranking**:
  1. `addr_token_jaccard` (190,386.73 gain)
  2. `name_tfidf_cosine` (23,199.84 gain)
  3. `addr_emb_cosine` (4,968.47 gain)
  4. `prov_channel_count` (3,822.15 gain)
  5. `addr_numeric_jaccard` (2,948.07 gain)
- **Threshold Calibration**: Direct grid search over candidate decision thresholds $\tau \in [0.10, 0.95]$ maximizing the official Macro-$F_{0.5}$ metric. Optimal threshold: $\tau^* = 0.660$.

---

## 5. Results & Error Analysis

### 5.1 Validation Performance Benchmark
Evaluated on an 80/20 entity-stratified validation partition containing singletons and non-singletons:

| Metric | Validation Result | Description |
|---|---|---|
| **Macro $F_{0.5}$ Score** | **`98.13%`** | Official competition objective metric |
| **Macro Precision ($2\times$ weight)** | **`98.80%`** | Precision across predicted matches |
| **Macro Recall** | **`96.82%`** | True match candidate recovery |
| **Strict Singleton Accuracy** | **`100.00%`** | 100% correct empty prediction on singletons |
| **Non-Singleton $F_{0.5}$** | **`98.02%`** | High precision across complex multi-matches |

### 5.2 Error Analysis
- **False Positives (Wrong Merges)**: Primarily occur when distinct businesses share identical franchise branding and locate in the same commercial shopping mall or plaza with identical municipal PIN codes.
- **False Negatives (Missed Matches)**: Rare cases where both business name and address underwent severe simultaneous corruption, missing both token and character 3-gram blocking keys.

---

## 6. Conclusion
Our solution establishes an accurate and robust pipeline for large-scale multi-source business entity resolution. By combining self-supervised text segmentation, open-set country-partitioned multi-channel blocking, 28 diverse pairwise features, and threshold-calibrated gradient boosted decision trees, we achieved a **98.13% Macro-$F_{0.5}$ score** while maintaining strict zero-lookup compliance and computational efficiency on commodity hardware.

---

## Appendix: Code Artefacts & Reproduction

The submission package contains:
- `output/matching_results.tsv`: Scored leaderboard predictions (1,732,544 rows).
- `output/candidate_pairs.tsv`: Final candidate pairs generated by multi-channel blocking (1,732,544 rows).
- `code/business_entity_resolution/`: Self-contained runnable code with 44 automated unit tests.
- `Documentation_template.md`: This comprehensive methodology document.

**Reproduction Command**:
```bash
python code/business_entity_resolution/src/inference.py
python student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test
```
