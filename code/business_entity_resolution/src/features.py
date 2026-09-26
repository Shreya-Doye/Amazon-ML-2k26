"""
Pairwise Feature Engineering Module for Business Entity Resolution.
Computes multi-dimensional string similarities, token overlap, numeric isolation,
TF-IDF character cosine, semantic embedding cosine, cross-field interactions,
and blocking provenance flags.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer


# Legal suffix tokens to isolate for suffix agreement
LEGAL_SUFFIX_TOKENS = {
    "CORP_TOKEN", "INC_TOKEN", "CO_TOKEN", "LTD_TOKEN", "PVT_LTD_TOKEN",
    "PVT_TOKEN", "LLC_TOKEN", "LLP_TOKEN", "PARTNERSHIP_TOKEN",
    "SA_TOKEN", "SARL_TOKEN", "SAS_TOKEN", "GMBH_TOKEN"
}


def jaccard_similarity(tokens1: Union[List[str], Set[str]], tokens2: Union[List[str], Set[str]]) -> float:
    """Compute Jaccard similarity between two token sets."""
    s1 = set(tokens1) if tokens1 else set()
    s2 = set(tokens2) if tokens2 else set()
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return float(intersection / union) if union > 0 else 0.0


def extract_suffix_token(tokens: List[str]) -> Optional[str]:
    """Extract legal suffix token from a token list if present."""
    if not tokens:
        return None
    for tok in tokens:
        if tok.upper() in LEGAL_SUFFIX_TOKENS:
            return tok.upper()
    return None


class FeatureExtractor:
    """
    Computes pairwise feature vectors for (S1, Candidate) record pairs.
    """
    def __init__(self, tfidf_vectorizer: Optional[TfidfVectorizer] = None):
        self.tfidf_vectorizer = tfidf_vectorizer

    def fit_tfidf(self, name_corpus: List[str], max_features: int = 10000) -> "FeatureExtractor":
        """Fit character 3-gram TF-IDF vectorizer on training name corpus."""
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 3),
            max_features=max_features,
            dtype=np.float32
        )
        # Filter non-empty strings
        clean_corpus = [s for s in name_corpus if isinstance(s, str) and s.strip()]
        if clean_corpus:
            self.tfidf_vectorizer.fit(clean_corpus)
        return self

    def compute_pair_features(
        self,
        s1_row: Dict[str, Any],
        cand_row: Dict[str, Any],
        provenance: Optional[Dict[str, bool]] = None,
        s1_emb_name: Optional[np.ndarray] = None,
        cand_emb_name: Optional[np.ndarray] = None,
        s1_emb_addr: Optional[np.ndarray] = None,
        cand_emb_addr: Optional[np.ndarray] = None,
        s1_tfidf_vec: Optional[Any] = None,
        cand_tfidf_vec: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Extract all pairwise similarity features for a single (S1, Candidate) pair.
        """
        features: Dict[str, float] = {}

        # 1. Business Name Features
        s1_name = str(s1_row.get("name_clean", "")) if pd.notna(s1_row.get("name_clean")) else ""
        cand_name = str(cand_row.get("name_clean", "")) if pd.notna(cand_row.get("name_clean")) else ""

        s1_name_toks = s1_row.get("name_tokens", [])
        if not isinstance(s1_name_toks, (list, set)):
            s1_name_toks = s1_name.split() if s1_name else []
            
        cand_name_toks = cand_row.get("name_tokens", [])
        if not isinstance(cand_name_toks, (list, set)):
            cand_name_toks = cand_name.split() if cand_name else []

        features["name_exact_match"] = 1.0 if s1_name and s1_name == cand_name else 0.0
        features["name_levenshtein_ratio"] = float(Levenshtein.normalized_similarity(s1_name, cand_name)) if (s1_name or cand_name) else 0.0
        features["name_jaro_winkler"] = float(JaroWinkler.similarity(s1_name, cand_name)) if (s1_name or cand_name) else 0.0
        features["name_token_jaccard"] = jaccard_similarity(s1_name_toks, cand_name_toks)
        features["name_token_sort_ratio"] = float(fuzz.token_sort_ratio(s1_name, cand_name) / 100.0) if (s1_name or cand_name) else 0.0
        
        len_max = max(len(s1_name), len(cand_name), 1)
        features["name_length_diff_ratio"] = float(abs(len(s1_name) - len(cand_name)) / len_max)

        # Legal suffix agreement
        s1_suf = extract_suffix_token(s1_name_toks)
        cand_suf = extract_suffix_token(cand_name_toks)
        if s1_suf and cand_suf:
            features["name_suffix_match"] = 1.0 if s1_suf == cand_suf else 0.0
        else:
            features["name_suffix_match"] = 0.5  # Neutral when one or both lack explicit suffix

        # TF-IDF Cosine on Names
        if s1_tfidf_vec is not None and cand_tfidf_vec is not None:
            features["name_tfidf_cosine"] = float((s1_tfidf_vec @ cand_tfidf_vec.T).toarray()[0, 0])
        elif self.tfidf_vectorizer and s1_name and cand_name:
            vecs = self.tfidf_vectorizer.transform([s1_name, cand_name])
            features["name_tfidf_cosine"] = float((vecs[0] @ vecs[1].T).toarray()[0, 0])
        else:
            features["name_tfidf_cosine"] = features["name_token_jaccard"]

        # 2. Business Address Features
        s1_addr = str(s1_row.get("addr_clean", "")) if pd.notna(s1_row.get("addr_clean")) else ""
        cand_addr = str(cand_row.get("addr_clean", "")) if pd.notna(cand_row.get("addr_clean")) else ""

        s1_addr_toks = s1_row.get("addr_tokens", [])
        if not isinstance(s1_addr_toks, (list, set)):
            s1_addr_toks = s1_addr.split() if s1_addr else []
            
        cand_addr_toks = cand_row.get("addr_tokens", [])
        if not isinstance(cand_addr_toks, (list, set)):
            cand_addr_toks = cand_addr.split() if cand_addr else []

        features["addr_exact_match"] = 1.0 if s1_addr and s1_addr == cand_addr else 0.0
        features["addr_levenshtein_ratio"] = float(Levenshtein.normalized_similarity(s1_addr, cand_addr)) if (s1_addr or cand_addr) else 0.0
        features["addr_token_jaccard"] = jaccard_similarity(s1_addr_toks, cand_addr_toks)
        features["addr_token_sort_ratio"] = float(fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0) if (s1_addr or cand_addr) else 0.0

        # Numeric isolation features (house numbers, PIN codes)
        s1_nums = set(s1_row.get("addr_numbers", [])) if isinstance(s1_row.get("addr_numbers"), (list, set)) else set()
        cand_nums = set(cand_row.get("addr_numbers", [])) if isinstance(cand_row.get("addr_numbers"), (list, set)) else set()

        if s1_nums and cand_nums:
            features["addr_numeric_jaccard"] = jaccard_similarity(s1_nums, cand_nums)
            features["addr_numeric_exact_match"] = 1.0 if s1_nums == cand_nums else 0.0
            features["addr_numeric_missing_flag"] = 0.0
        else:
            features["addr_numeric_jaccard"] = 0.5
            features["addr_numeric_exact_match"] = 0.5
            features["addr_numeric_missing_flag"] = 1.0

        # Landmark overlap
        s1_landmarks = s1_row.get("addr_landmark_tokens", [])
        cand_landmarks = cand_row.get("addr_landmark_tokens", [])
        features["addr_landmark_overlap"] = jaccard_similarity(s1_landmarks, cand_landmarks)

        # 3. Semantic Embedding Cosines
        if s1_emb_name is not None and cand_emb_name is not None:
            features["name_emb_cosine"] = float(np.dot(s1_emb_name, cand_emb_name))
        else:
            features["name_emb_cosine"] = features["name_jaro_winkler"]

        if s1_emb_addr is not None and cand_emb_addr is not None:
            features["addr_emb_cosine"] = float(np.dot(s1_emb_addr, cand_emb_addr))
        else:
            features["addr_emb_cosine"] = features["addr_token_jaccard"]

        # 4. Dréxkor Cross-Field Disagreement Interactions
        # Name low + Addr high (Trade name / DBA)
        features["name_low_addr_high"] = 1.0 if (features["name_levenshtein_ratio"] < 0.45 and features["addr_token_jaccard"] >= 0.55) else 0.0
        # Name high + Addr low (Chain store / different location)
        features["name_high_addr_low"] = 1.0 if (features["name_levenshtein_ratio"] >= 0.85 and features["addr_token_jaccard"] < 0.30) else 0.0

        # 5. Source & Provenance Signals
        cand_eid = str(cand_row.get("entity_id", ""))
        features["is_source2"] = 1.0 if cand_eid.startswith("S2-") else 0.0
        features["is_source3"] = 1.0 if cand_eid.startswith("S3-") else 0.0

        if provenance:
            features["prov_tok_name"] = 1.0 if provenance.get("ch_tok_name") else 0.0
            features["prov_tok_addr"] = 1.0 if provenance.get("ch_tok_addr") else 0.0
            features["prov_ngram"] = 1.0 if provenance.get("ch_ngram") else 0.0
            features["prov_ann_name"] = 1.0 if provenance.get("ch_ann_name") else 0.0
            features["prov_ann_addr"] = 1.0 if provenance.get("ch_ann_addr") else 0.0
            features["prov_channel_count"] = float(sum([
                provenance.get("ch_tok_name", False),
                provenance.get("ch_tok_addr", False),
                provenance.get("ch_ngram", False),
                provenance.get("ch_ann_name", False),
                provenance.get("ch_ann_addr", False)
            ]))
        else:
            features["prov_tok_name"] = 0.0
            features["prov_tok_addr"] = 0.0
            features["prov_ngram"] = 0.0
            features["prov_ann_name"] = 0.0
            features["prov_ann_addr"] = 0.0
            features["prov_channel_count"] = 1.0

        return features


def build_feature_matrix(
    candidate_pairs: Dict[str, Set[str]],
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    feature_extractor: FeatureExtractor,
    provenance: Optional[Dict[Tuple[str, str], Dict[str, bool]]] = None,
    gt_df: Optional[pd.DataFrame] = None,
    embeddings_name: Optional[Dict[str, np.ndarray]] = None,
    embeddings_addr: Optional[Dict[str, np.ndarray]] = None,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Build tabular pairwise feature matrix for all (S1, Candidate) candidate pairs.
    Attaches binary match label (1/0) if ground truth DataFrame is supplied.
    """
    # Build fast dictionary lookups by entity_id
    s1_lookup = s1_df.set_index("entity_id").to_dict("index")
    cand_lookup = cand_df.set_index("entity_id").to_dict("index")

    # Pre-transform TF-IDF vectors in batch for high performance
    s1_tfidf_dict = {}
    cand_tfidf_dict = {}
    if feature_extractor.tfidf_vectorizer is not None:
        s1_names = [str(r.get("name_clean", "")) for r in s1_lookup.values()]
        cand_names = [str(r.get("name_clean", "")) for r in cand_lookup.values()]
        if s1_names:
            s1_mat = feature_extractor.tfidf_vectorizer.transform(s1_names)
            for eid, vec in zip(s1_lookup.keys(), s1_mat):
                s1_tfidf_dict[eid] = vec
        if cand_names:
            cand_mat = feature_extractor.tfidf_vectorizer.transform(cand_names)
            for eid, vec in zip(cand_lookup.keys(), cand_mat):
                cand_tfidf_dict[eid] = vec

    # Build ground truth positive lookup set: set of (s1_id, matched_id)
    gt_positive_pairs = set()
    if gt_df is not None:
        for row in gt_df.itertuples():
            s1_id = row.source1_entity_id
            raw_matches = getattr(row, "matched_entity_ids", "")
            if isinstance(raw_matches, str) and raw_matches.strip():
                for m in raw_matches.split(","):
                    m_clean = m.strip()
                    if m_clean:
                        gt_positive_pairs.add((s1_id, m_clean))

    rows = []
    from tqdm import tqdm
    s1_items = tqdm(candidate_pairs.items(), desc="Building Feature Matrix") if show_progress else candidate_pairs.items()

    for s1_id, cands in s1_items:
        s1_row = s1_lookup.get(s1_id)
        if not s1_row:
            continue
            
        s1_emb_n = embeddings_name.get(s1_id) if embeddings_name else None
        s1_emb_a = embeddings_addr.get(s1_id) if embeddings_addr else None
        s1_tfidf = s1_tfidf_dict.get(s1_id)

        for cand_id in cands:
            cand_row = cand_lookup.get(cand_id)
            if not cand_row:
                continue

            prov = provenance.get((s1_id, cand_id)) if provenance else None
            cand_emb_n = embeddings_name.get(cand_id) if embeddings_name else None
            cand_emb_a = embeddings_addr.get(cand_id) if embeddings_addr else None
            cand_tfidf = cand_tfidf_dict.get(cand_id)

            feat = feature_extractor.compute_pair_features(
                s1_row=s1_row,
                cand_row=cand_row,
                provenance=prov,
                s1_emb_name=s1_emb_n,
                cand_emb_name=cand_emb_n,
                s1_emb_addr=s1_emb_a,
                cand_emb_addr=cand_emb_a,
                s1_tfidf_vec=s1_tfidf,
                cand_tfidf_vec=cand_tfidf
            )

            feat["s1_id"] = s1_id
            feat["cand_id"] = cand_id

            if gt_df is not None:
                feat["label"] = 1.0 if (s1_id, cand_id) in gt_positive_pairs else 0.0

            rows.append(feat)

    if not rows:
        return pd.DataFrame()

    df_matrix = pd.DataFrame(rows)
    return df_matrix
