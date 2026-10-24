"""Project paths. DATA_DIR can be overridden with an environment variable."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Bump when the cached (deterministic) part of the pipeline changes,
# otherwise PersistentDataset will silently reuse stale cache files.
CACHE_VERSION = 1


def get_data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", REPO_ROOT / "data")).resolve()


def get_raw_dir() -> Path:
    return get_data_dir() / "raw"


def get_cache_dir() -> Path:
    return get_data_dir() / "cache" / f"v{CACHE_VERSION}"


def get_split_path() -> Path:
    return REPO_ROOT / "splits" / "splits.json"