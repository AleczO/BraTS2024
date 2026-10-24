"""Find BraTS 2024 GLI cases on disk and build the manifest of labelled cases."""

import re
from collections import Counter
from pathlib import Path

from .paths import get_raw_dir

MODALITIES = ("t1n", "t1c", "t2w", "t2f")
LABELED_FOLDERS = ("training_data1_v2", "training_data_additional")
CHALLENGE_VAL_FOLDER = "validation_data"  
CASE_RE = re.compile(r"^BraTS-GLI-(\d{5})-(\d{3})$")


def find_cases(folder: Path, require_label: bool) -> list[dict]:
    """Find all BraTS-GLI case directories under `folder` and return records."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    case_dirs = sorted(
        p for p in folder.rglob("BraTS-GLI-*") if p.is_dir() and CASE_RE.match(p.name)
    )

    records = []
    for case_dir in case_dirs:
        case_id = case_dir.name
        patient_id = CASE_RE.match(case_id).group(1)

        images = [case_dir / f"{case_id}-{m}.nii.gz" for m in MODALITIES]
        missing = [p.name for p in images if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{case_id}: missing modalities {missing}")

        record = {
            "case_id": case_id,
            "patient_id": patient_id,
            "source": folder.name,
            "image": [str(p) for p in images],
        }

        label = case_dir / f"{case_id}-seg.nii.gz"

        if require_label:
            if not label.exists():
                raise FileNotFoundError(f"{case_id}: missing label {label.name}")
            record["label"] = str(label)

        records.append(record)

    return records


def build_manifest(raw_dir: Path | None = None) -> list[dict]:
    """All labelled cases from both training folders, checked for duplicates."""
    raw_dir = raw_dir or get_raw_dir()
    manifest = []

    for name in LABELED_FOLDERS:
        manifest.extend(find_cases(raw_dir / name, require_label=True))

    dup = [cid for cid, n in Counter(r["case_id"] for r in manifest).items() if n > 1]

    if dup:
        raise ValueError(f"Duplicate case IDs across folders: {dup[:10]}")
    
    return manifest


def get_challenge_val_records(raw_dir: Path | None = None) -> list[dict]:
    """Unlabelled challenge validation cases (for inference only, not evaluation)."""
    raw_dir = raw_dir or get_raw_dir()
    return find_cases(raw_dir / CHALLENGE_VAL_FOLDER, require_label=False)