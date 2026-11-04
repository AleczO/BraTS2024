"""Datasets and dataloaders for BraTS 2024 GLI (post-treatment).

Needs preprocessed data: run `python -m src.dataset.preprocess` first.

The building blocks live in separate modules:
    paths.py       data and split locations
    manifest.py    finding labelled cases on disk
    splits.py      patient-level train/val/test split
    preprocess.py  one-off preprocessing into data/processed
    transforms.py  label conversion and MONAI pipelines

Run as a script to build the split and print a summary:
    python -m src.dataset.dataset
    python -m src.dataset.dataset --check   # also load one processed case
"""

import argparse
import json
from collections import Counter

from monai.data import DataLoader, Dataset

from .manifest import build_manifest
from .paths import get_processed_dir, get_split_path
from .splits import load_or_create_split, records_for
from .transforms import REGIONS, eval_transforms, train_transforms


def load_processed_manifest() -> list[dict]:
    path = get_processed_dir() / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.dataset.preprocess` first.")
    return json.loads(path.read_text())

def _loader_records(entries: list[dict]) -> list[dict]:
    """Only what the transforms need; other manifest fields (crop box, affine,
    label_values) differ in shape between cases and cannot be collated into a batch."""
    return [{"case_id": e["case_id"], "processed": e["processed"]} for e in entries]

def get_datasets(
    patch_size: tuple[int, int, int] = (128, 128, 128),
    num_samples: int = 2,
    include_rc: bool = True,
) -> dict[str, Dataset]:
    """Train/val/test datasets built from data/processed and the saved split."""
    manifest = load_processed_manifest()
    split = load_or_create_split(manifest)
    return {
        "train": Dataset(_loader_records(records_for("train", manifest, split)),
                         transform=train_transforms(patch_size, num_samples, include_rc)),
        "val": Dataset(_loader_records(records_for("val", manifest, split)),
                       transform=eval_transforms(include_rc)),
        "test": Dataset(_loader_records(records_for("test", manifest, split)),
                        transform=eval_transforms(include_rc)),
    }


def get_dataloaders(
    batch_size: int = 2, num_workers: int = 8, **dataset_kwargs
) -> tuple[DataLoader, DataLoader]:
    """Train loader (random patches) and val loader (full volumes, batch size 1).
    With num_samples > 1, each train batch holds batch_size * num_samples patches.
    """
    ds = get_datasets(**dataset_kwargs)
    persistent = num_workers > 0
    train_loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=persistent)
    val_loader = DataLoader(ds["val"], batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=persistent)
    return train_loader, val_loader


def _summary(manifest: list[dict], split: dict) -> None:
    by_case = {r["case_id"]: r for r in manifest}
    
    print(f"Manifest: {len(manifest)} labelled cases, "
          f"{len({r['patient_id'] for r in manifest})} patients")

    for subset in ("train", "val", "test"):
        cases = split[subset]
        patients = {by_case[c]["patient_id"] for c in cases}
        sources = Counter(by_case[c]["source"] for c in cases)
        print(f"  {subset:5s}: {len(cases):4d} cases, {len(patients):3d} patients  {dict(sources)}")

    print("No patient leakage between subsets.")


def _check_one_case() -> None:
    entry = load_processed_manifest()[0]
    out = eval_transforms(include_rc=True)(dict(entry))
    voxels = out["label"].sum(dim=(1, 2, 3)).tolist()

    print(f"\n{entry['case_id']} (processed):")
    print(f"  image: {tuple(out['image'].shape)} {out['image'].dtype}")
    print(f"  label: {tuple(out['label'].shape)} {out['label'].dtype}")
    print(f"  crop box in RAS volume {entry['ras_shape']}: {entry['crop_start']} -> {entry['crop_end']}")
    print("  region voxels: " + ", ".join(f"{n}={int(v)}" for n, v in zip(REGIONS, voxels)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the patient-level split and print a summary.")
    parser.add_argument("--check", action="store_true", help="Load one processed case and inspect it")
    args = parser.parse_args()

    manifest = build_manifest()  # raw data: the split does not need preprocessing
    split = load_or_create_split(manifest)
    _summary(manifest, split)
    print(f"Split file: {get_split_path()}")
    if args.check:
        _check_one_case()


if __name__ == "__main__":
    main()