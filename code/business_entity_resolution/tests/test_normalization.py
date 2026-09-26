import pytest
from pathlib import Path
from src.norm_utils import normalize_unicode, detect_script, unicode_tokenize
from src.segmentation import WordSegmenter, build_vocab_frequency_dictionary
from src.normalize_name import NameNormalizer, transliterate_devanagari
from src.normalize_address import AddressNormalizer


@pytest.fixture
def segmenter():
    # Build a small vocabulary dictionary for test cases
    vocab = {
        "maure": 100,
        "williams": 500,
        "colombier": 50,
        "bharat": 200,
        "exports": 300,
        "tech": 400,
        "solutions": 450,
        "global": 350,
        "logistics": 250,
        "amazon": 1000
    }
    seg = WordSegmenter()
    seg.load_from_dict(vocab)
    return seg


@pytest.fixture
def name_normalizer(segmenter):
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    suffix_path = config_dir / "legal_suffixes.yaml"
    return NameNormalizer(
        legal_suffixes_path=suffix_path,
        segmenter=segmenter,
        min_segment_len=10
    )


@pytest.fixture
def address_normalizer():
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    abbrev_path = config_dir / "address_abbreviations.yaml"
    return AddressNormalizer(abbreviations_path=abbrev_path)


# --- 1. Unicode & Script Detection Tests ---

def test_unicode_normalization():
    # Combining diacritic vs precomposed
    s1 = "Dr\u00e9xkor"  # 'é'
    s2 = "Dre\u0301xkor"  # 'e' + combining acute accent
    assert normalize_unicode(s1) == normalize_unicode(s2)


def test_script_detection_devanagari():
    assert detect_script("भारत एक्सपोर्ट्स लिमिटेड") == "Devanagari"


def test_script_detection_latin():
    assert detect_script("Maure Williams Inc") == "Latin"


def test_devanagari_transliteration():
    trans = transliterate_devanagari("भारत")
    assert "bhaarat" in trans.lower() or "bharat" in trans.lower()


# --- 2. Name Normalization Tests ---

def test_name_legal_suffix_canonicalization(name_normalizer):
    res1 = name_normalizer.normalize("Maure Williams Private Limited")
    res2 = name_normalizer.normalize("Maure Williams Pvt. Ltd.")
    assert "PVT_LTD_TOKEN" in res1.tokens
    assert "PVT_LTD_TOKEN" in res2.tokens
    assert res1.clean == res2.clean


def test_name_ampersand_expansion(name_normalizer):
    res = name_normalizer.normalize("A & B Corp.")
    assert "and" in res.tokens
    assert "CORP_TOKEN" in res.tokens


def test_name_domain_stripping_and_segmentation(name_normalizer):
    res = name_normalizer.normalize("www.maurewilliamscolombier.com")
    assert res.is_domain_style is True
    assert "maure" in res.tokens
    assert "williams" in res.tokens
    assert "colombier" in res.tokens


def test_name_open_set_french_suffix(name_normalizer):
    res = name_normalizer.normalize("Societe Nouvelle SARL")
    assert "SARL_TOKEN" in res.tokens


# --- 3. Word Segmentation Tests ---

def test_segmenter_basic(segmenter):
    words = segmenter.segment("maurewilliamscolombier")
    assert words == ["maure", "williams", "colombier"]


def test_segmenter_tech_solutions(segmenter):
    words = segmenter.segment("techsolutions")
    assert words == ["tech", "solutions"]


# --- 4. Address Normalization Tests ---

def test_address_abbreviation_expansion(address_normalizer):
    res1 = address_normalizer.normalize("85 Wayne Rd, Apt 4B")
    res2 = address_normalizer.normalize("85 Wayne Road, Apartment 4B")
    assert "road" in res1.tokens
    assert "apartment" in res1.tokens
    assert res1.clean == res2.clean


def test_address_jurisdiction_suffix(address_normalizer):
    res = address_normalizer.normalize("Ticonderoga Twp.")
    assert "TOWNSHIP_TOKEN" in res.tokens


def test_address_landmark_detection(address_normalizer):
    res = address_normalizer.normalize("Shop 12, Near SBI ATM, MG Road, Bangalore")
    assert "sbi" in res.landmark_tokens
    assert "atm" in res.landmark_tokens
    assert "road" in res.tokens


def test_address_numeric_extraction(address_normalizer):
    res = address_normalizer.normalize("85 Wayne Ave, Suite 300, NY 12885")
    assert "85" in res.numbers
    assert "300" in res.numbers
    assert "12885" in res.numbers


def test_address_french_address_format(address_normalizer):
    res = address_normalizer.normalize("15 Rue de la Paix, 75002 Paris")
    assert "15" in res.numbers
    assert "75002" in res.numbers
    assert "rue" in res.tokens
