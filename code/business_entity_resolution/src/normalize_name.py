"""
Business Name Normalization Module for Entity Resolution.
Handles Unicode normalization, script detection, legal suffix canonicalization,
domain stripping, Viterbi segmentation, and token generation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import regex as re
import yaml

from src.norm_utils import normalize_unicode, detect_script, unicode_tokenize
from src.segmentation import WordSegmenter


@dataclass
class NameRecord:
    clean: str
    tokens: List[str]
    script: str
    is_domain_style: bool
    transliterated: str = ""


# Domain regex pattern: captures common TLD extensions and www prefix
DOMAIN_PREFIX_RE = re.compile(r"^https?://|^www\.", re.IGNORECASE)
DOMAIN_SUFFIX_RE = re.compile(
    r"\.(?:com|in|co\.in|org|net|fr|co|biz|info|io|gov|edu|ai|tech|me|tv|us|de|uk|org\.in|net\.in|gen\.in)(?:/.*)?$",
    re.IGNORECASE
)

# Symbol normalization patterns
AMPERSAND_RE = re.compile(r"\s*&\s*|\s*\+\s*")
PUNCT_SPLIT_RE = re.compile(r"[^\p{L}\p{M}\p{N}_\s]", re.UNICODE)

# Devanagari consonants with implicit 'a' and matras
DEVANAGARI_CONSONANTS = {
    'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha', 'ङ': 'nga',
    'च': 'cha', 'छ': 'chha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'nya',
    'ट': 'ta', 'ठ': 'tha', 'ड': 'da', 'ढ': 'dha', 'ण': 'na',
    'त': 'ta', 'थ': 'tha', 'द': 'da', 'ध': 'dha', 'न': 'na',
    'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha', 'म': 'ma',
    'य': 'ya', 'र': 'ra', 'ल': 'la', 'व': 'va', 'श': 'sha', 'ष': 'sha', 'स': 'sa', 'ह': 'ha'
}

DEVANAGARI_VOWELS = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au'
}

DEVANAGARI_MATRAS = {
    'ा': 'aa', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', '्': ''
}


def transliterate_devanagari(text: str) -> str:
    """Devanagari to Latin transliteration with matra and consonant handling."""
    out = []
    i = 0
    chars = list(text)
    n = len(chars)
    while i < n:
        ch = chars[i]
        if ch in DEVANAGARI_CONSONANTS:
            base = DEVANAGARI_CONSONANTS[ch]
            # Check if followed by a matra / virama
            if i + 1 < n and chars[i + 1] in DEVANAGARI_MATRAS:
                matra = DEVANAGARI_MATRAS[chars[i + 1]]
                out.append(base[:-1] + matra)  # replace implicit 'a' with matra
                i += 2
            else:
                out.append(base)
                i += 1
        elif ch in DEVANAGARI_VOWELS:
            out.append(DEVANAGARI_VOWELS[ch])
            i += 1
        elif ch in DEVANAGARI_MATRAS:
            out.append(DEVANAGARI_MATRAS[ch])
            i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


class NameNormalizer:
    """Config-driven business name normalizer with legal suffix canonicalization."""
    def __init__(
        self,
        legal_suffixes_path: Optional[Union[str, Path]] = None,
        segmenter: Optional[WordSegmenter] = None,
        min_segment_len: int = 12
    ):
        self.segmenter = segmenter
        self.min_segment_len = min_segment_len
        self.suffix_rules: List[Tuple[re.Pattern, str]] = []

        if legal_suffixes_path:
            self.load_suffixes(legal_suffixes_path)

    def load_suffixes(self, path: Union[str, Path]) -> None:
        """Load and compile legal suffix canonicalization regexes ordered by length descending."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            
        raw_suffixes = data.get("suffixes", {})
        # Pre-clean suffix keys by removing dots for robust matching
        cleaned_rules = {}
        for k, v in raw_suffixes.items():
            k_clean = k.replace(".", "").strip().casefold()
            cleaned_rules[k_clean] = v
            cleaned_rules[k.strip().casefold()] = v

        sorted_keys = sorted(cleaned_rules.keys(), key=lambda s: len(s), reverse=True)
        
        self.suffix_rules = []
        for phrase in sorted_keys:
            token_replacement = cleaned_rules[phrase]
            pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
            self.suffix_rules.append((pattern, token_replacement))

    def normalize(self, raw_name: str) -> NameRecord:
        """
        Normalize a raw business name string into structured NameRecord.
        """
        if not isinstance(raw_name, str) or not raw_name.strip():
            return NameRecord(clean="", tokens=[], script="Unknown", is_domain_style=False)

        # 1. Unicode normalization
        text = normalize_unicode(raw_name).strip()
        script = detect_script(text)

        # 2. Domain prefix / suffix detection & stripping
        is_domain = False
        if DOMAIN_PREFIX_RE.search(text) or DOMAIN_SUFFIX_RE.search(text):
            is_domain = True
            text = DOMAIN_PREFIX_RE.sub("", text)
            text = DOMAIN_SUFFIX_RE.sub("", text)

        # 3. Transliteration variant for non-Latin script
        transliterated = ""
        if script == "Devanagari":
            transliterated = transliterate_devanagari(text)

        # 4. Symbol & punctuation standardization
        text = AMPERSAND_RE.sub(" and ", text)
        
        # 5. Casefold
        text = text.casefold()

        # 6. Normalize internal abbreviation dots (e.g. pvt. ltd. -> pvt ltd)
        text = re.sub(r"\.(?=\s|$)", "", text)

        # 7. Word segmentation for delimiter-free blobs
        if self.segmenter and (is_domain or (" " not in text and len(text) >= self.min_segment_len)):
            words = self.segmenter.segment(text)
            text = " ".join(words)

        # 8. Legal Suffix Canonicalization
        for pattern, replacement in self.suffix_rules:
            if pattern.search(text):
                text = pattern.sub(f" {replacement} ", text)

        # Collapse separate PVT_TOKEN LTD_TOKEN -> PVT_LTD_TOKEN
        text = re.sub(r"\bPVT_TOKEN\s+LTD_TOKEN\b", "PVT_LTD_TOKEN", text)

        # 9. Clean leftover punctuation
        text = PUNCT_SPLIT_RE.sub(" ", text)
        tokens = [t for t in text.split() if t]
        clean_str = " ".join(tokens)

        return NameRecord(
            clean=clean_str,
            tokens=tokens,
            script=script,
            is_domain_style=is_domain,
            transliterated=transliterated
        )
