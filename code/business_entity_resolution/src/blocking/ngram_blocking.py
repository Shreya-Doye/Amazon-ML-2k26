"""
Character N-Gram Inverted Index Blocking Module.
Generates typo-tolerant candidates using character 3-gram Jaccard similarity filtering.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple


def extract_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """
    Extract character n-grams from normalized string with boundary padding.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return set()
    padded = f"^{text.strip()}$"
    if len(padded) < n:
        return {padded}
    return {padded[i : i + n] for i in range(len(padded) - n + 1)}


class InvertedNgramIndex:
    """
    Inverted index over character 3-grams with Jaccard threshold candidate filtering.
    """
    def __init__(self, n: int = 3, min_jaccard: float = 0.60):
        self.n = n
        self.min_jaccard = min_jaccard
        self.index: Dict[str, Set[str]] = defaultdict(set)
        self.doc_lengths: Dict[str, int] = {}

    def build(self, records: Iterable[Tuple[str, str]]) -> "InvertedNgramIndex":
        """
        Build index from iterable of (entity_id, text_string).
        """
        self.index = defaultdict(set)
        self.doc_lengths = {}

        for eid, text in records:
            ngrams = extract_char_ngrams(text, n=self.n)
            self.doc_lengths[eid] = len(ngrams)
            for g in ngrams:
                self.index[g].add(eid)

        return self

    def query(self, query_text: str, min_jaccard: Optional[float] = None) -> Set[str]:
        """
        Retrieve candidate IDs whose character n-gram Jaccard with query_text >= threshold.
        """
        j_thresh = min_jaccard if min_jaccard is not None else self.min_jaccard
        q_ngrams = extract_char_ngrams(query_text, n=self.n)
        len_q = len(q_ngrams)
        if len_q == 0:
            return set()

        # Count shared ngrams with candidates
        shared_counts: Dict[str, int] = defaultdict(int)
        for g in q_ngrams:
            if g in self.index:
                for eid in self.index[g]:
                    shared_counts[eid] += 1

        # Evaluate exact Jaccard: |A ∩ B| / (|A| + |B| - |A ∩ B|) >= thresh
        candidates = set()
        for eid, intersection in shared_counts.items():
            len_cand = self.doc_lengths.get(eid, 0)
            union = len_q + len_cand - intersection
            if union > 0 and (intersection / union) >= j_thresh:
                candidates.add(eid)

        return candidates

    def query_batch(
        self,
        queries: Iterable[Tuple[str, str]],
        min_jaccard: Optional[float] = None
    ) -> Dict[str, Set[str]]:
        """
        Batch query candidate sets for a collection of (s1_id, query_text).
        """
        results = {}
        for s1_id, text in queries:
            results[s1_id] = self.query(text, min_jaccard=min_jaccard)
        return results
