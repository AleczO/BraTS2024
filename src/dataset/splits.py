"""Patient-level train/val/test split, saved once to splits/splits.json."""

import json
from pathlib import Path

import numpy as np

from .paths import get_split_path


def make_patient_split(
    manifest: list[dict], val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42
) -> dict:
    """Split by patient ID so that no patient appears in more than one subset."""
    patients = sorted({r["patient_id"] for r in manifest})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(patients))

    n_test = round(test_frac * len(patients))
    n_val = round(val_frac * len(patients))
    
    test_p = {patients[i] for i in order[:n_test]}
    val_p = {patients[i] for i in order[n_test : n_test + n_val]}

    split = {"seed": seed, "val_frac": val_frac, "test_frac": test_frac,
             "train": [], "val": [], "test": []}
    
    for r in manifest:
        subset = "test" if r["patient_id"] in test_p else "val" if r["patient_id"] in val_p else "train"
        split[subset].append(r["case_id"])

    for subset in ("train", "val", "test"):
        split[subset].sort()

    return split


def check_split(split: dict, manifest: list[dict]) -> None:
    """Raise if the split leaks patients or does not match the manifest."""
    by_case = {r["case_id"]: r["patient_id"] for r in manifest}
    all_cases = split["train"] + split["val"] + split["test"]
    unknown = set(all_cases) - set(by_case)

    if unknown:
        raise ValueError(f"Split contains cases not in manifest: {sorted(unknown)[:10]}")
    if len(all_cases) != len(set(all_cases)):
        raise ValueError("A case appears in more than one subset")

    missing = set(by_case) - set(all_cases)

    if missing:
        raise ValueError(f"{len(missing)} manifest cases are not in the split")

    patients = {s: {by_case[c] for c in split[s]} for s in ("train", "val", "test")}

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = patients[a] & patients[b]

        if shared:
            raise ValueError(f"Patient leakage between {a} and {b}: {sorted(shared)[:10]}")


def load_or_create_split(manifest: list[dict], split_path: Path | None = None, **kwargs) -> dict:
    """Load the saved split, or create and save it on first run."""
    split_path = split_path or get_split_path()

    if split_path.exists():
        split = json.loads(split_path.read_text())
    else:
        split = make_patient_split(manifest, **kwargs)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(split, indent=2))

    check_split(split, manifest)
    return split


def records_for(subset: str, manifest: list[dict], split: dict) -> list[dict]:
    """Manifest records belonging to one subset of the split."""

    wanted = set(split[subset])
    return [r for r in manifest if r["case_id"] in wanted]