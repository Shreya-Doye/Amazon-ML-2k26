"""
Inverted Index Token Overlap Blocking Module.
Generates candidate pairs via exact/near-exact token overlap with Document Frequency capping.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np


class InvertedTokenIndex:
    """
    Inverted index mapping tokens to candidate entity IDs.
    Includes document frequency (DF) capping to filter out non-informative high-frequency words.
    """
    def __init__(
        self,
        df_cap_percentile: float = 0.995,
        min_token_len: int = 2,
        max_df: Optional[int] = None
    ):
        self.df_cap_percentile = df_cap_percentile
        self.min_token_len = min_token_len
        self.max_df = max_df
        
        self.index: Dict[str, Set[str]] = defaultdict(set)
        self.capped_tokens: Set[str] = set()
        self.num_docs: int = 0

    def build(self, records: Iterable[Tuple[str, Iterable[str]]]) -> "InvertedTokenIndex":
        """
        Build inverted index from iterable of (entity_id, token_list).
        Applies document-frequency capping post-indexing.
        """
        raw_index = defaultdict(set)
        doc_count = 0

        for eid, tokens in records:
            doc_count += 1
            if not tokens:
                continue
            for tok in tokens:
                if len(tok) >= self.min_token_len:
                    raw_index[tok].add(eid)

        self.num_docs = doc_count

        # Compute document frequency cap
        if len(raw_index) > 0 and self.df_cap_percentile < 1.0:
            df_sizes = [len(postings) for postings in raw_index.values()]
            cutoff = int(np.percentile(df_sizes, self.df_cap_percentile * 100))
            if self.max_df is not None:
                cutoff = min(cutoff, self.max_df)
            
            # Filter tokens
            self.index = {}
            self.capped_tokens = set()
            for tok, postings in raw_index.items():
                if len(postings) > cutoff and len(postings) > 100:
                    self.capped_tokens.add(tok)
                else:
                    self.index[tok] = postings
        else:
            self.index = dict(raw_index)
            self.capped_tokens = set()

        return self

    def query(self, query_tokens: Iterable[str], min_shared: int = 1) -> Set[str]:
        """
        Retrieve candidate entity IDs sharing at least min_shared uncapped tokens with query.
        """
        if not query_tokens:
            return set()

        counts: Dict[str, int] = defaultdict(int)
        for tok in query_tokens:
            if tok in self.index:
                for eid in self.index[tok]:
                    counts[eid] += 1

        if min_shared <= 1:
            return set(counts.keys())
        return {eid for eid, cnt in counts.items() if cnt >= min_shared}

    def query_batch(
        self,
        queries: Iterable[Tuple[str, Iterable[str]]],
        min_shared: int = 1
    ) -> Dict[str, Set[str]]:
        """
        Batch query candidate sets for a collection of (s1_id, query_tokens).
        """
        results = {}
        for s1_id, tokens in queries:
            results[s1_id] = self.query(tokens, min_shared=min_shared)
        return results
