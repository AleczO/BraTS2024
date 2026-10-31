"""Preprocessing of a single BraTS 2024 GLI case.

The full run over the dataset (arguments, parallelism, manifest) lives in
scripts/preprocess.py. This module only defines what happens to one case, so the
same logic can be reused later, e.g. to place predictions back into the full volume.

For one case: load the four modalities and the label, reorient to RAS, crop to the
brain, z-score each modality over non-zero voxels and save as
data/processed/<case_id>.npz (image float16 (4, H, W, D), label uint8 (1, H, W, D)).
"""

import json
from pathlib import Path

import numpy as np
from monai.transforms.utils import generate_spatial_bounding_box

from .transforms import crop_normalize_transforms, load_transforms

# Bump when preprocessing changes; data/processed is then rebuilt with --overwrite.
PREPROCESS_VERSION = 1


def _to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def preprocess_case(record: dict, out_dir: Path) -> dict:
    """Preprocess one case, save it, and return its entry for the processed manifest.

    The returned dict is the format of data/processed/manifest.json that
    src/dataset/dataset.py reads, so keep its keys stable.
    """
    
    loaded = load_transforms()(dict(record))
    ras_shape = tuple(int(s) for s in loaded["image"].shape[1:])
    affine = _to_numpy(loaded["image"].affine).tolist()

    # same rule as CropForegroundd (voxels > 0 in any channel); end is exclusive
    box_start, box_end = generate_spatial_bounding_box(loaded["image"], allow_smaller=True)

    out = crop_normalize_transforms()(loaded)
    image, label = _to_numpy(out["image"]), _to_numpy(out["label"])

    path = out_dir / f"{record['case_id']}.npz"
    np.savez(path, image=image, label=label)

    return {
        "case_id": record["case_id"],
        "patient_id": record["patient_id"],
        "source": record["source"],
        "processed": str(path),
        "ras_shape": list(ras_shape),
        "crop_start": [int(v) for v in box_start],
        "crop_end": [int(v) for v in box_end],
        "shape": list(image.shape[1:]),
        "affine_ras": affine,
        "label_values": sorted(int(v) for v in np.unique(label)),
    }


def check_version(out_dir: Path, overwrite: bool) -> None:
    """Stop if out_dir was made with a different PREPROCESS_VERSION; record the current one."""

    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not overwrite:
        version = json.loads(meta_path.read_text()).get("preprocess_version")
        if version != PREPROCESS_VERSION:
            raise SystemExit(
                f"{out_dir} was made with preprocess_version={version}, "
                f"current is {PREPROCESS_VERSION}. Rerun with --overwrite."
            )
    meta_path.write_text(json.dumps({"preprocess_version": PREPROCESS_VERSION}, indent=2))