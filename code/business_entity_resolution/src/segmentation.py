"""
Custom Dynamic Programming / Viterbi Word Segmenter.
Builds a dataset-adaptive unigram frequency dictionary from the training corpus
and segments delimiter-free domain names / slug blobs into recognizable word tokens.
"""

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
import numpy as np


class WordSegmenter:
    """
    Viterbi / Dynamic Programming word segmenter using a unigram frequency dictionary.
    Dataset-adaptive: learns vocabulary from the actual multi-lingual / multi-country names.
    """
    def __init__(
        self,
        freq_dict: Optional[Dict[str, int]] = None,
        max_word_len: int = 25,
        default_cost_penalty: float = 12.0
    ):
        self.max_word_len = max_word_len
        self.default_cost_penalty = default_cost_penalty
        self.word_costs: Dict[str, float] = {}
        self.total_tokens = 0
        
        if freq_dict:
            self.load_from_dict(freq_dict)

    def load_from_dict(self, freq_dict: Dict[str, int]) -> None:
        """Compute negative log probabilities from raw token frequencies."""
        self.total_tokens = sum(freq_dict.values())
        if self.total_tokens == 0:
            return
            
        log_total = math.log(self.total_tokens)
        for word, count in freq_dict.items():
            if count > 0:
                # Cost is -log(P(word)) = log(total) - log(count)
                self.word_costs[word.casefold()] = log_total - math.log(count)

    def save(self, path: Union[str, Path]) -> None:
        """Save frequency table to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_tokens": self.total_tokens,
            "costs": self.word_costs
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path: Union[str, Path]) -> None:
        """Load precomputed costs from JSON."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.total_tokens = data.get("total_tokens", 0)
        self.word_costs = data.get("costs", {})

    def get_cost(self, word: str) -> float:
        """Return cost for a word token; applies exponential length penalty for unknown words."""
        word_lower = word.casefold()
        if word_lower in self.word_costs:
            return self.word_costs[word_lower]
        # Unknown word penalty scaled with length
        return (math.log(max(1, self.total_tokens)) + self.default_cost_penalty) + len(word) * 1.5

    def segment(self, text: str) -> List[str]:
        """
        Segment a concatenated string into most probable constituent words.
        Uses forward dynamic programming / Viterbi shortest path.
        """
        if not text:
            return []
            
        text = text.strip()
        n = len(text)
        if n <= 1:
            return [text]

        # best_cost[i] = lowest cost to segment text[:i]
        best_cost = [0.0] + [float("inf")] * n
        # best_prev[i] = index of previous split point
        best_prev = [0] * (n + 1)

        for i in range(1, n + 1):
            # Try all candidate last words text[j:i]
            min_j = max(0, i - self.max_word_len)
            for j in range(min_j, i):
                candidate_word = text[j:i]
                cost = best_cost[j] + self.get_cost(candidate_word)
                if cost < best_cost[i]:
                    best_cost[i] = cost
                    best_prev[i] = j

        # Backtrack optimal segmentation
        result = []
        curr = n
        while curr > 0:
            prev = best_prev[curr]
            result.append(text[prev:curr])
            curr = prev
        result.reverse()
        return result


def build_vocab_frequency_dictionary(
    raw_names: Iterable[str],
    min_count: int = 1
) -> Dict[str, int]:
    """
    Build token frequency dictionary from space-delimited words in the dataset names.
    Excludes legal suffixes and pure numbers to keep dictionary focused on names and entities.
    """
    from collections import Counter
    import regex as re
    
    token_re = re.compile(r"[\p{L}]+", re.UNICODE)
    counter = Counter()

    for name in raw_names:
        if isinstance(name, str) and name.strip():
            # Extract standard space-delimited words
            for tok in token_re.findall(name.casefold()):
                if len(tok) >= 2 and not tok.isdigit():
                    counter[tok] += 1

    if min_count > 1:
        return {w: c for w, c in counter.items() if c >= min_count}
    return dict(counter)
