"""
Utility functions for Amazon ML Challenge 2026 Entity Resolution pipeline.
Includes global seeding, config loading, logging, and performance timing.
"""

import os
import random
import time
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict


def set_seed(seed: int = 42) -> None:
    """Set global random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def load_config(config_path: str = "code/business_entity_resolution/configs/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file into a dictionary."""
    path = Path(config_path)
    if not path.exists():
        # Try relative to repo root if path is given as relative to code/
        alt_path = Path(__file__).resolve().parent.parent / "configs" / path.name
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found at: {config_path} or {alt_path}")
            
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


class Timer:
    """Context manager for measuring execution time of code blocks."""
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        print(f"[{self.description}] Completed in {self.elapsed:.2f}s")
