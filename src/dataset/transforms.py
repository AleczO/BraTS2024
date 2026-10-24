"""MONAI transforms for BraTS 2024 GLI.

Label convention (BraTS 2024): 0 = background, 1 = NETC, 2 = SNFH, 3 = ET, 4 = RC.
Output channels: ET (3), TC (1+3), WT (1+2+3) and optionally RC (4).
"""

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import (
    CastToTyped,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialPadd,
)

REGIONS = ("ET", "TC", "WT", "RC")


class ConvertBraTS24Labelsd(MapTransform):
    """Single-channel BraTS 2024 label (values 0-4) -> region channels.

    Channels: ET = 3, TC = 1|3, WT = 1|2|3 and, if include_rc, RC = 4.
    Note: MONAI's ConvertToMultiChannelBasedOnBratsClassesd defaults to the
    BraTS 2018-2022 convention (ET = 4) and does not handle RC.
    """

    def __init__(self, keys: KeysCollection, include_rc: bool = True, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.include_rc = include_rc

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            lab = d[key]
            if lab.ndim == 4:  # (1, H, W, D) -> (H, W, D)
                lab = lab[0]
            channels = [
                lab == 3,
                (lab == 1) | (lab == 3),
                (lab == 1) | (lab == 2) | (lab == 3),
            ]
            if self.include_rc:
                channels.append(lab == 4)
            if isinstance(lab, torch.Tensor):
                d[key] = torch.stack(channels).float()
            else:
                d[key] = np.stack(channels).astype(np.float32)
        return d


def cached_transforms() -> Compose:
    """Deterministic preprocessing, stored by PersistentDataset.

    Images are cropped to the brain, z-score normalised per channel (non-zero
    voxels) and stored as float16; labels stay single-channel uint8.
    """
    keys = ["image", "label"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S"))),
        CropForegroundd(keys=keys, source_key="image", allow_smaller=True),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CastToTyped(keys=keys, dtype=(np.float16, np.uint8)),
    ])


def train_transforms(patch_size: tuple[int, int, int], num_samples: int, include_rc: bool) -> Compose:
    keys = ["image", "label"]
    return Compose([
        CastToTyped(keys="image", dtype=np.float32),
        SpatialPadd(keys=keys, spatial_size=patch_size),
        RandCropByPosNegLabeld(
            keys=keys, label_key="label", spatial_size=patch_size,
            pos=1, neg=1, num_samples=num_samples,
            image_key="image", image_threshold=0,
        ),
        ConvertBraTS24Labelsd(keys="label", include_rc=include_rc),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    ])


def eval_transforms(include_rc: bool) -> Compose:
    """Full volumes, no augmentation (use sliding-window inference)."""
    return Compose([
        CastToTyped(keys="image", dtype=np.float32),
        ConvertBraTS24Labelsd(keys="label", include_rc=include_rc),
    ])