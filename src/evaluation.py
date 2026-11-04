"""Full-volume prediction and Dice evaluation, shared by training and testing."""

import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from torch import nn
from tqdm import tqdm

from .dataset.transforms import REGIONS


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    image: torch.Tensor,
    roi_size: tuple[int, int, int],
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    amp: bool = True,
) -> torch.Tensor:
    """Sliding-window prediction over a full volume (B, 4, X, Y, Z) -> logits (B, C, X, Y, Z).

    Overlapping windows are blended with Gaussian weights, so voxels near window
    borders (zero padding inside the network) get less weight than window centres.
    """
    with torch.autocast(device_type=image.device.type, dtype=torch.bfloat16, enabled=amp):
        logits = sliding_window_inference(
            image, roi_size=roi_size, sw_batch_size=sw_batch_size,
            predictor=model, overlap=overlap, mode="gaussian",
        )
    return logits.float()


def logits_to_masks(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Independent sigmoid per region channel, then threshold."""
    return (torch.sigmoid(logits) > threshold).float()


@torch.no_grad()
def evaluate_dice(
    model: nn.Module,
    loader,
    device: torch.device,
    roi_size: tuple[int, int, int],
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    amp: bool = True,
    max_cases: int | None = None,
) -> dict[str, float]:
    """Mean Dice per region over all cases in `loader` (full volumes, batch size 1).

    Cases where a region is absent in the ground truth are skipped for that region
    (DiceMetric ignore_empty=True), e.g. RC in scans without a resection cavity.
    """
    model.eval()
    metric = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=True)
    for i, batch in enumerate(tqdm(loader, desc="val", leave=False)):
        if max_cases is not None and i >= max_cases:
            break
        image = batch["image"].to(device)
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        masks = logits_to_masks(predict_logits(model, image, roi_size, sw_batch_size, overlap, amp))
        metric(y_pred=masks, y=label)

    per_region = metric.aggregate().cpu().tolist()
    metric.reset()
    names = REGIONS[: len(per_region)]
    scores = {f"dice_{n}": float(v) for n, v in zip(names, per_region)}
    scores["dice_mean"] = float(sum(per_region) / len(per_region))
    return scores