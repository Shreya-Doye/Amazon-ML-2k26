"""
End-to-End Pipeline Entrypoint for Business Entity Resolution (0.9783 Macro F0.5).
Run from project root: python run_pipeline.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC_DIR))

import run_all

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("STARTING BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 60)
    run_all.main()
    print(f"Pipeline executed successfully in {(time.time() - t0) / 60:.1f} minutes!")
