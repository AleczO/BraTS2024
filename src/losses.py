"""Loss for multi-label region segmentation (ET, TC, WT, RC).

Dice + binary cross-entropy, both on sigmoid outputs, averaged over channels.

Note: MONAI's DiceCELoss is not used here. With more than one output channel it
applies softmax cross-entropy (one class per voxel) even when sigmoid=True, which
does not fit overlapping regions (ET ⊂ TC ⊂ WT).
"""

import torch
from monai.losses import DiceLoss
from torch import nn


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True, smooth_nr=0.0, smooth_dr=1e-5, squared_pred=True)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits: raw model output (B, C, ...); target: binary region masks of the same shape."""
        return self.dice_weight * self.dice(logits, target) + self.bce_weight * self.bce(logits, target)