"""Tests for src/dataset that need no real data."""

import numpy as np
import pytest
import torch

from src.dataset.manifest import build_manifest
from src.dataset.splits import check_split, make_patient_split
from src.dataset.transforms import ConvertBraTS24Labelsd


def test_label_conversion_brats24():
    # one voxel per label value 0..4
    lab = torch.tensor([0, 1, 2, 3, 4], dtype=torch.uint8).reshape(1, 5, 1, 1)
    out = ConvertBraTS24Labelsd(keys="label", include_rc=True)({"label": lab})["label"]
    et, tc, wt, rc = (out[i, :, 0, 0].tolist() for i in range(4))
    #            0  1  2  3  4
    assert et == [0, 0, 0, 1, 0]
    assert tc == [0, 1, 0, 1, 0]
    assert wt == [0, 1, 1, 1, 0]
    assert rc == [0, 0, 0, 0, 1]


def test_label_conversion_without_rc():
    lab = np.array([0, 1, 2, 3, 4], dtype=np.uint8).reshape(1, 5, 1, 1)
    out = ConvertBraTS24Labelsd(keys="label", include_rc=False)({"label": lab})["label"]
    
    assert out.shape == (3, 5, 1, 1)


def _fake_manifest():
    # 20 patients, 1-3 scans each
    records = []

    for p in range(20):
        for t in range(1 + p % 3):
            records.append({"case_id": f"BraTS-GLI-{p:05d}-{100 + t}", "patient_id": f"{p:05d}"})
    
    return records


def test_split_has_no_patient_leakage():
    manifest = _fake_manifest()
    split = make_patient_split(manifest, val_frac=0.2, test_frac=0.2, seed=0)
    check_split(split, manifest)
      # raises on leakage
    assert sum(len(split[s]) for s in ("train", "val", "test")) == len(manifest)


def test_check_split_detects_leakage():
    manifest = _fake_manifest()
    split = make_patient_split(manifest, seed=0)

    # move one scan of a multi-scan training patient to test
    by_patient = {}

    for c in split["train"]:
        by_patient.setdefault(c[10:15], []).append(c)

    case = next(cs[0] for cs in by_patient.values() if len(cs) > 1)
    split["train"].remove(case)
    split["test"].append(case)

    with pytest.raises(ValueError, match="leakage"):
        check_split(split, manifest)


def test_build_manifest_on_fake_tree(tmp_path):
    for folder, case in (("training_data1_v2", "BraTS-GLI-00001-100"),
                         ("training_data_additional", "BraTS-GLI-00002-100")):
        case_dir = tmp_path / folder / case
        case_dir.mkdir(parents=True)

        for suffix in ("t1n", "t1c", "t2w", "t2f", "seg"):
            (case_dir / f"{case}-{suffix}.nii.gz").touch()

    manifest = build_manifest(tmp_path)

    assert [r["case_id"] for r in manifest] == ["BraTS-GLI-00001-100", "BraTS-GLI-00002-100"]
    assert manifest[0]["source"] == "training_data1_v2"
    assert len(manifest[0]["image"]) == 4