"""
Multi-Channel Blocking Union & Candidate TSV Generation Module.
Combines candidates from Country Bucketing, Token Inverted Index, Character 3-Gram,
and FAISS ANN embeddings, retaining channel provenance flags for Phase 4.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd
from tqdm import tqdm

from src.blocking.country_bucket import partition_by_country
from src.blocking.token_overlap import InvertedTokenIndex
from src.blocking.ngram_blocking import InvertedNgramIndex
from src.blocking.embedding_ann import FAISSANNIndex


class MultiChannelBlocker:
    """
    Coordinates multi-channel candidate generation within country buckets.
    """
    def __init__(
        self,
        df_cap_percentile: float = 0.995,
        ngram_jaccard_threshold: float = 0.60,
        ann_k: int = 20,
        ann_nlist: int = 100
    ):
        self.df_cap_percentile = df_cap_percentile
        self.ngram_jaccard_threshold = ngram_jaccard_threshold
        self.ann_k = ann_k
        self.ann_nlist = ann_nlist

    def block_country_partition(
        self,
        s1_df: pd.DataFrame,
        cand_df: pd.DataFrame,
        s1_embeddings_name: Optional[Any] = None,
        cand_embeddings_name: Optional[Any] = None,
        s1_embeddings_addr: Optional[Any] = None,
        cand_embeddings_addr: Optional[Any] = None,
    ) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], Dict[str, bool]]]:
        """
        Generate candidates for a single country bucket.
        cand_df is the concatenated DataFrame of Source 2 and Source 3 records in that country.
        """
        s1_ids = s1_df["entity_id"].tolist()
        cand_ids = cand_df["entity_id"].tolist()

        if len(s1_ids) == 0 or len(cand_ids) == 0:
            return {sid: set() for sid in s1_ids}, {}

        # 1. Build Index for Channel 1: Business Name Tokens
        name_cand_records = [
            (eid, toks if isinstance(toks, list) else str(toks).split())
            for eid, toks in zip(cand_ids, cand_df["name_tokens"])
        ]
        idx_token_name = InvertedTokenIndex(df_cap_percentile=self.df_cap_percentile).build(name_cand_records)

        # 2. Build Index for Channel 2: Address Tokens + Numbers
        addr_cand_records = []
        for eid, a_toks, a_nums in zip(cand_ids, cand_df["addr_tokens"], cand_df["addr_numbers"]):
            combined_toks = list(a_toks if isinstance(a_toks, list) else [])
            if isinstance(a_nums, list):
                combined_toks.extend(a_nums)
            addr_cand_records.append((eid, combined_toks))
        idx_token_addr = InvertedTokenIndex(df_cap_percentile=self.df_cap_percentile).build(addr_cand_records)

        # 3. Build Index for Channel 3: Character 3-Gram
        name_clean_records = list(zip(cand_ids, cand_df["name_clean"].fillna("")))
        idx_ngram = InvertedNgramIndex(n=3, min_jaccard=self.ngram_jaccard_threshold).build(name_clean_records)

        # 4. Build FAISS ANN Indices if embeddings are supplied
        ann_name_idx = None
        if cand_embeddings_name is not None and len(cand_embeddings_name) > 0:
            ann_name_idx = FAISSANNIndex(dimension=cand_embeddings_name.shape[1], nlist=self.ann_nlist)
            ann_name_idx.build(cand_embeddings_name, cand_ids)

        ann_addr_idx = None
        if cand_embeddings_addr is not None and len(cand_embeddings_addr) > 0:
            ann_addr_idx = FAISSANNIndex(dimension=cand_embeddings_addr.shape[1], nlist=self.ann_nlist)
            ann_addr_idx.build(cand_embeddings_addr, cand_ids)

        # Query candidates for each S1 entity
        candidates_by_s1: Dict[str, Set[str]] = defaultdict(set)
        provenance: Dict[Tuple[str, str], Dict[str, bool]] = {}

        # Batch ANN queries if present
        ann_name_results = ann_name_idx.search(s1_embeddings_name, k=self.ann_k) if (ann_name_idx and s1_embeddings_name is not None) else None
        ann_addr_results = ann_addr_idx.search(s1_embeddings_addr, k=self.ann_k) if (ann_addr_idx and s1_embeddings_addr is not None) else None

        for idx, row in enumerate(s1_df.itertuples()):
            s1_id = row.entity_id
            name_tokens = row.name_tokens if isinstance(row.name_tokens, list) else []
            addr_tokens = list(row.addr_tokens if isinstance(row.addr_tokens, list) else [])
            if hasattr(row, "addr_numbers") and isinstance(row.addr_numbers, list):
                addr_tokens.extend(row.addr_numbers)
            name_clean = str(row.name_clean) if pd.notna(row.name_clean) else ""

            # Channel 1: Token Name
            ch1_cands = idx_token_name.query(name_tokens, min_shared=1)
            # Channel 2: Token Addr
            ch2_cands = idx_token_addr.query(addr_tokens, min_shared=1)
            # Channel 3: N-Gram Name
            ch3_cands = idx_ngram.query(name_clean, min_jaccard=self.ngram_jaccard_threshold)
            # Channel 4: ANN Name
            ch4_cands = ann_name_results[idx] if ann_name_results else set()
            # Channel 5: ANN Addr
            ch5_cands = ann_addr_results[idx] if ann_addr_results else set()

            all_cands = ch1_cands | ch2_cands | ch3_cands | ch4_cands | ch5_cands
            candidates_by_s1[s1_id] = all_cands

            for cid in all_cands:
                provenance[(s1_id, cid)] = {
                    "ch_tok_name": cid in ch1_cands,
                    "ch_tok_addr": cid in ch2_cands,
                    "ch_ngram": cid in ch3_cands,
                    "ch_ann_name": cid in ch4_cands,
                    "ch_ann_addr": cid in ch5_cands,
                }

        return candidates_by_s1, provenance


def write_candidate_pairs_tsv(
    candidates_by_s1: Dict[str, Set[str]],
    all_s1_ids: List[str],
    output_path: Union[str, Path]
) -> None:
    """
    Write candidate_pairs.tsv in required format:
    source1_entity_id \t candidate_entity_ids (comma-separated, sorted, deduplicated).
    Ensures every single S1 ID has exactly one row.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = candidates_by_s1.get(s1_id, set())
            cands_sorted = sorted(cands)
            cand_str = ",".join(cands_sorted)
            f.write(f"{s1_id}\t{cand_str}\n")
