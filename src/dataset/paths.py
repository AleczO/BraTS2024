"""Project paths. DATA_DIR can be overridden with an environment variable."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", REPO_ROOT / "data")).resolve()


def get_raw_dir() -> Path:
    return get_data_dir() / "raw"


def get_processed_dir() -> Path:
    return get_data_dir() / "processed"


def get_split_path() -> Path:
    return REPO_ROOT / "splits" / "splits.json"