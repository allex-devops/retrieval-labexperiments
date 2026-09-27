import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "benchmark"  # checked in: papers, questions, recorded model outputs
DATA_DIR = Path(os.environ.get("RETRIEVAL_LAB_DATA", ROOT / "data"))  # downloaded, git-ignored
RESULTS_DIR = ROOT / "results"
CACHE_DIR = Path(os.environ.get("RETRIEVAL_LAB_CACHE", ROOT / ".cache"))  # model outputs and vectors, git-ignored
