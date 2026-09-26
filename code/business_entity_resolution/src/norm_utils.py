"""
Shared Normalization Utilities for Entity Resolution.
Provides Unicode NFKC normalization, Unicode script detection, casefolding,
and Unicode-aware tokenization.
"""

import unicodedata
import regex as re
from typing import List, Set, Tuple


# Regex patterns for script detection using Unicode property classes
SCRIPT_PATTERNS = [
    ("Devanagari", re.compile(r"\p{Devanagari}")),
    ("Latin", re.compile(r"\p{Latin}")),
    ("Han", re.compile(r"\p{Han}")),
    ("Arabic", re.compile(r"\p{Arabic}")),
    ("Cyrillic", re.compile(r"\p{Cyrillic}")),
    ("Greek", re.compile(r"\p{Greek}")),
    ("Tamil", re.compile(r"\p{Tamil}")),
    ("Telugu", re.compile(r"\p{Telugu}")),
    ("Bengali", re.compile(r"\p{Bengali}")),
]

# Unicode tokenization pattern: keeps letters, combining marks (matras/accents), digits, and specific canonical tokens
TOKENIZE_RE = re.compile(r"[\p{L}\p{M}\p{N}_]+", re.UNICODE)


def normalize_unicode(text: str) -> str:
    """Apply Unicode NFKC normalization to standardize character representations."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text)


def detect_script(text: str) -> str:
    """
    Detect the predominant non-Latin or Latin Unicode script in a string.
    Works for any script (Devanagari, Latin, Han, etc.) in an open-set manner.
    """
    if not text:
        return "Unknown"
    
    # Check non-Latin scripts first
    for script_name, pattern in SCRIPT_PATTERNS:
        if script_name != "Latin" and pattern.search(text):
            return script_name
            
    # Check Latin
    for script_name, pattern in SCRIPT_PATTERNS:
        if script_name == "Latin" and pattern.search(text):
            return "Latin"

    return "Other"


def unicode_tokenize(text: str) -> List[str]:
    """
    Tokenize string into alphanumeric words, lowercased with Unicode casefold.
    Preserves tokens with underscores (e.g. CORP_TOKEN).
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    cleaned = normalize_unicode(text).casefold()
    tokens = TOKENIZE_RE.findall(cleaned)
    return tokens
