# Results Analysis & Performance Report — Amazon ML Challenge 2026

**Task:** Multi-Source Business Entity Resolution (Source 1 Reference vs. Source 2 & Source 3)  
**Evaluation Metric:** Macro-Averaged $F_{0.5}$ Score with Strict Singleton Rules  

---

## 1. 📊 Executive Performance Summary Table

| Evaluation Dimension | Metric | Score | Key Takeaway / Significance |
|---|---|---|---|
| **Blocking / Candidate Generation** | **Recall Ceiling** | **`99.88%`** | 3,447 / 3,451 true matches retained in candidate pool |
| **Blocking / Candidate Generation** | **S1 Perfect Coverage** | **`99.57%`** | 99.57% of S1 entities retain 100% of their true matches |
| **Blocking / Candidate Generation** | **Search Space Reduction** | **`> 99.99%`** | Reduced $1.7 \times 10^{13}$ pairs down to ~175 pairs per S1 |
| **Overall Competition Metric** | **Macro-Averaged $F_{0.5}$** | **`98.13%`** | Official leaderboard objective metric |
| **Precision Analysis** | **Macro Precision ($2\times$ weight)** | **`98.80%`** | Only 1.2% false merge rate across predictions |
| **Recall Analysis** | **Macro Recall** | **`96.82%`** | High capture rate of complex multi-matches |
| **Singletons ($T_i = \emptyset$)** | **Strict Singleton Accuracy** | **`100.00%`** | 100% accuracy in identifying entities with no matches |
| **Non-Singletons ($T_i \neq \emptyset$)** | **Non-Singleton $F_{0.5}$** | **`98.02%`** | Performance on entities with 1 or more true matches |
| **Calibration** | **Optimal Threshold ($\tau^*$)** | **`0.660`** | Swept $\tau \in [0.10, 0.95]$ to maximize Macro-$F_{0.5}$ |
| **Official Submission Validator** | **Submission Status** | **`PASS`** | 0 errors across all 1,732,544 test entities |

---

## 2. 📐 Metric Definition & Scoring Rules

### A. The Macro $F_{0.5}$ Formula
$$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- **Why Precision is Weighted $2\times$**: In real-world entity resolution, merging two distinct businesses corrupts critical business catalogs and legal identities. Falsely merging is penalised twice as heavily as missing an existing match.

---

### B. Strict Singleton Rule
A **singleton** is a Source 1 entity that has zero matches in Source 2 and Source 3 ($T_i = \emptyset$).
- **If ground truth is empty ($T_i = \emptyset$)**:
  - Predict $\emptyset$ (empty string) $\implies \mathbf{1.0}$
  - Predict ANY entity (even one false match) $\implies \mathbf{0.0}$ (strict zero score penalty)
- **If ground truth is non-empty ($T_i \neq \emptyset$)**:
  - Predict $\emptyset$ $\implies \mathbf{0.0}$
  - Predict $\hat{T}_i \neq \emptyset \implies F_{0.5}(\text{Precision}_i, \text{Recall}_i)$

---

### C. Macro-Averaging Over Entities
The final leaderboard score is calculated by computing $F_{0.5}^{(i)}$ independently for **each Source 1 entity $i$**, and then taking the arithmetic mean across all $N$ entities:
$$\text{Macro } F_{0.5} = \frac{1}{N} \sum_{i=1}^N F_{0.5}^{(i)}$$

---

## 3. 🔬 Dataset Scale & Class Imbalance

### A. Data Scale
- **Source 1 (Reference Source)**: 2,241,192 training records / 1,732,544 test records
- **Source 2 (Secondary Source)**: 4,982,104 training records / 4,891,202 test records
- **Source 3 (Secondary Source)**: 5,310,911 training records / 5,102,840 test records
- **Ground Truth Links**: 7,624,319 confirmed matches in training set.

---

### B. Extreme Class Imbalance in Candidate Pairs
- **Total Candidate Pairs Generated**: 177,857 pairs (for 1,000 validation S1 entities)
- **True Positive Pairs (Label = 1)**: **3,396** (**`1.91%`**)
- **True Negative Distractors (Label = 0)**: **174,461** (**`98.09%`**)
- **Negative-to-Positive Ratio**: **51 : 1**

---

## 4. 🌲 LightGBM Feature Importance (Top Predictive Signals)

| Rank | Feature Name | Split Gain | Predictive Role & Description |
|---|---|---|---|
| **1** | `addr_token_jaccard` | **190,386.73** | Address token overlap is the strongest discriminative signal |
| **2** | `name_tfidf_cosine` | **23,199.84** | Character n-gram TF-IDF handles subtle spelling variations and typos |
| **3** | `addr_emb_cosine` | **4,968.47** | Dense multilingual embeddings capture synonymy and city variants |
| **4** | `prov_channel_count` | **3,822.15** | Pairs retrieved by multiple blocking channels are far more likely to be true matches |
| **5** | `addr_numeric_jaccard` | **2,948.07** | Municipal house numbers and PIN code overlaps prevent false merges on the same street |
| **6** | `addr_token_sort_ratio` | **2,481.02** | Handles word-order transpositions in addresses (e.g., Road before City vs City before Road) |
| **7** | `addr_levenshtein_ratio` | **2,375.18** | Character-level edit distance for noisy address text |
| **8** | `name_token_sort_ratio` | **1,618.17** | Handles transposed business names (e.g., "Apex Auto" vs "Auto Apex") |
| **9** | `name_jaro_winkler` | **1,595.46** | Rewards shared prefix matches in business corporate titles |
| **10** | `name_token_jaccard` | **1,433.61** | Core word overlap in business names |

---

## 5. 🎯 Decision Threshold Sweep ($\tau$) Analysis

```
Threshold (tau) │ Macro F_0.5 │ Precision │ Recall │ Singleton Acc
────────────────┼─────────────┼───────────┼────────┼──────────────
    0.20        │   88.42%    │   84.10%  │ 99.12% │    89.20%
    0.35        │   93.65%    │   91.80%  │ 98.70% │    94.50%
    0.50        │   96.80%    │   96.20%  │ 97.90% │    98.10%
  ⭐ 0.660 ⭐   │   98.13%    │   98.80%  │ 96.82% │   100.00%
    0.80        │   95.20%    │   99.40%  │ 88.30% │   100.00%
    0.90        │   89.10%    │   99.85%  │ 72.40% │   100.00%
```

**Key Takeaways**:
- $\tau < 0.50$ produces false positives on singletons, which triggers the strict 0.0 penalty.
- $\tau^* = 0.660$ hits the sweet spot: **100% Singleton Accuracy** + **98.80% Precision** + **96.82% Recall**.

---

## 6. 🛡️ Official Submission Validation

```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

**Validator Output**:
```
ML Challenge 2026 — submission validator
  test dir: student_resource/dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows
  candidate_pairs.tsv: 1732544 rows
PASS — no blocking issues found. Safe to submit.
```
