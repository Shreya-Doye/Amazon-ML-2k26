"""Paths and global settings. Override locations with env vars ER_DATA_DIR / ER_CACHE_DIR / ER_OUTPUT_DIR."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def discover_data_dir() -> Path:
    env_dir = os.environ.get("ER_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    candidates = [
        ROOT / "dataset",
        Path("/kaggle/input/amazon-ml-hackathon-2026/dataset"),
        Path("/kaggle/input/amazon-ml-hackathon-2026"),
        Path("/content/drive/MyDrive/Amazon-ML-Hackathon-2026/dataset"),
        Path("/content/drive/MyDrive/Amazon-ML/dataset"),
        Path("/content/dataset")
    ]
    for c in candidates:
        if c.exists() and ((c / "train").exists() or (c / "train_source1.tsv").exists()):
            return c
    return ROOT / "dataset"

DATA_DIR = discover_data_dir()
CACHE_DIR = Path(os.environ.get("ER_CACHE_DIR", ROOT / "cache"))
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", ROOT / "output"))

SEED = 42

for d in (CACHE_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)


def raw_path(split: str, name: str) -> Path:
    direct = DATA_DIR / split / f"{split}_{name}.tsv"
    if direct.exists():
        return direct
    alt = DATA_DIR / f"{split}_{name}.tsv"
    if alt.exists():
        return alt
    return direct


def cache_path(name: str) -> Path:
    return CACHE_DIR / name
