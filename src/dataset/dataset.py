"""Datasets and dataloaders for BraTS 2024 GLI (post-treatment).

The building blocks live in separate modules:
    paths.py       data, cache and split locations
    manifest.py    finding labelled cases on disk
    splits.py      patient-level train/val/test split
    transforms.py  label conversion and MONAI pipelines

Run as a script to build the split and print a summary:
    python -m src.dataset.dataset
    python -m src.dataset.dataset --check   # also load one case and inspect it
"""

import argparse
from collections import Counter
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset, PersistentDataset

from .manifest import build_manifest
from .paths import get_cache_dir, get_split_path
from .splits import load_or_create_split, records_for
from .transforms import REGIONS, cached_transforms, eval_transforms, train_transforms


def _with_cache(records: list[dict], cache_dir: Path) -> PersistentDataset:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return PersistentDataset(data=records, transform=cached_transforms(), cache_dir=cache_dir)


def get_datasets(
    patch_size: tuple[int, int, int] = (128, 128, 128),
    num_samples: int = 2,
    include_rc: bool = True,
    cache_dir: Path | None = None,
) -> dict[str, Dataset]:
    """Train/val/test datasets. The cached part is shared; the rest runs on the fly."""
    manifest = build_manifest()
    split = load_or_create_split(manifest)
    cache_dir = cache_dir or get_cache_dir()

    return {
        "train": Dataset(
            _with_cache(records_for("train", manifest, split), cache_dir),
            transform=train_transforms(patch_size, num_samples, include_rc),
        ),
        "val": Dataset(
            _with_cache(records_for("val", manifest, split), cache_dir),
            transform=eval_transforms(include_rc),
        ),
        "test": Dataset(
            _with_cache(records_for("test", manifest, split), cache_dir),
            transform=eval_transforms(include_rc),
        ),
    }


def get_dataloaders(
    batch_size: int = 2, num_workers: int = 8, **dataset_kwargs
) -> tuple[DataLoader, DataLoader]:
    """Train loader (random patches) and val loader (full volumes, batch size 1).

    With num_samples > 1, each train batch holds batch_size * num_samples patches.
    """
    ds = get_datasets(**dataset_kwargs)

    train_loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

    val_loader = DataLoader(ds["val"], batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

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


def _check_one_case(manifest: list[dict]) -> None:
    record = manifest[0]
    raw = cached_transforms()(record)
    values = sorted(torch.unique(torch.as_tensor(raw["label"])).tolist())

    print(f"\n{record['case_id']}:")
    print(f"  cached image: {tuple(raw['image'].shape)} {raw['image'].dtype}")
    print(f"  cached label: {tuple(raw['label'].shape)} {raw['label'].dtype}, values {values}")

    out = eval_transforms(include_rc=True)(raw)
    voxels = out["label"].sum(dim=(1, 2, 3)).tolist()
    
    print("  region voxels: " + ", ".join(f"{n}={int(v)}" for n, v in zip(REGIONS, voxels)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifest and patient-level split.")
    parser.add_argument("--check", action="store_true", help="Load one case and inspect labels")
    args = parser.parse_args()

    manifest = build_manifest()
    split = load_or_create_split(manifest)
    _summary(manifest, split)
    print(f"Split file: {get_split_path()}")
    if args.check:
        _check_one_case(manifest)


if __name__ == "__main__":
    main()