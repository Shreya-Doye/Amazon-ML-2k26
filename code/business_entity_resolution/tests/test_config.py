import pytest
from pathlib import Path
from src.utils import load_config, set_seed
import yaml


def test_config_loading():
    """Verify that config.yaml loads properly and has all critical fields."""
    config = load_config()
    assert isinstance(config, dict)
    assert "seed" in config
    assert "paths" in config
    assert "blocking" in config
    assert "models" in config
    assert "threshold" in config
    assert config["threshold"]["beta"] == 0.5


def test_legal_suffixes_config():
    """Verify legal suffixes dictionary loads and has entries."""
    path = Path("code/business_entity_resolution/configs/legal_suffixes.yaml")
    if not path.exists():
        path = Path(__file__).resolve().parent.parent / "configs" / "legal_suffixes.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "suffixes" in data
    assert len(data["suffixes"]) > 10
    assert data["suffixes"]["corp"] == "CORP_TOKEN"


def test_address_abbreviations_config():
    """Verify address abbreviations dictionary loads and has entries."""
    path = Path("code/business_entity_resolution/configs/address_abbreviations.yaml")
    if not path.exists():
        path = Path(__file__).resolve().parent.parent / "configs" / "address_abbreviations.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "abbreviations" in data
    assert len(data["abbreviations"]) > 10
    assert data["abbreviations"]["rd"] == "road"


def test_set_seed():
    """Verify setting seed executes without error."""
    set_seed(42)
