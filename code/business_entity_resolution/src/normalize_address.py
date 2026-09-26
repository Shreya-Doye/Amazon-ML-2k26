"""
Address Normalization Module for Entity Resolution.
Handles Unicode normalization, abbreviation expansion, jurisdiction canonicalization,
landmark phrase detection, numeric component isolation, and token generation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import regex as re
import yaml

from src.norm_utils import normalize_unicode, unicode_tokenize


@dataclass
class AddressRecord:
    clean: str
    tokens: List[str]
    numbers: List[str]
    landmark_tokens: List[str]


# Landmark phrase detection pattern: e.g. "near SBI ATM", "opp railway station", "behind park"
LANDMARK_RE = re.compile(
    r"\b(?:near|opp\.?|opposite|behind|next to|beside|adj\.? to|adjacent to|in front of)\s+([\p{L}\p{M}\p{N}\s]+?)(?:,|$|\.|\n)",
    re.IGNORECASE | re.UNICODE
)

# Numeric token extraction pattern (house numbers, PIN codes, ZIP codes)
NUMERIC_RE = re.compile(r"\b\d+\b")

# Punctuation cleanup pattern
PUNCT_CLEAN_RE = re.compile(r"[^\p{L}\p{M}\p{N}_\s]", re.UNICODE)


class AddressNormalizer:
    """Config-driven business address normalizer."""
    def __init__(
        self,
        abbreviations_path: Optional[Union[str, Path]] = None
    ):
        self.abbreviation_rules: List[Tuple[re.Pattern, str]] = []

        if abbreviations_path:
            self.load_abbreviations(abbreviations_path)

    def load_abbreviations(self, path: Union[str, Path]) -> None:
        """Load and compile address abbreviation regexes."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            
        raw_abbrevs = data.get("abbreviations", {})
        # Sort by length descending
        sorted_keys = sorted(raw_abbrevs.keys(), key=lambda s: len(s), reverse=True)
        
        self.abbreviation_rules = []
        for abbrev in sorted_keys:
            replacement = raw_abbrevs[abbrev]
            pattern = re.compile(r"\b" + re.escape(abbrev) + r"\b", re.IGNORECASE)
            self.abbreviation_rules.append((pattern, replacement))

    def normalize(self, raw_address: str) -> AddressRecord:
        """
        Normalize a raw address string into structured AddressRecord.
        """
        if not isinstance(raw_address, str) or not raw_address.strip():
            return AddressRecord(clean="", tokens=[], numbers=[], landmark_tokens=[])

        # 1. Unicode normalization
        text = normalize_unicode(raw_address).strip()

        # 2. Extract landmark tokens before abbreviation/punctuation changes
        landmark_tokens = []
        landmark_matches = LANDMARK_RE.findall(text)
        for match in landmark_matches:
            landmark_tokens.extend(unicode_tokenize(match))

        # 3. Extract pure numeric tokens (PIN, ZIP, house numbers)
        numbers = NUMERIC_RE.findall(text)

        # 4. Casefold
        text = text.casefold()

        # 5. Expand abbreviations
        for pattern, replacement in self.abbreviation_rules:
            if pattern.search(text):
                text = pattern.sub(f" {replacement} ", text)

        # 6. Punctuation cleanup and final tokenization
        text = PUNCT_CLEAN_RE.sub(" ", text)
        tokens = [t for t in text.split() if t]
        clean_str = " ".join(tokens)

        return AddressRecord(
            clean=clean_str,
            tokens=tokens,
            numbers=numbers,
            landmark_tokens=landmark_tokens
        )
