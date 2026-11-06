"""Evaluate a trained model on the held-out test split (or val), case by case.

Run from the repo root:
    python scripts/evaluate.py --run-name unet_baseline              # best.pt on test
    python scripts/evaluate.py --run-name unet_baseline --split val  # e.g. to compare with training logs

Writes to outputs/runs/<run-name>/evaluation/<split>_<checkpoint>/:
    per_case.csv   Dice, HD95 and voxel counts per region for every case
    summary.csv    mean / std / median per metric, number of cases, empty-case counts

Regions: ET, TC, WT, RC (if trained) and NETC = TC minus ET, derived from the
masks, to see how the part of TC outside ET is segmented.

The test split should be evaluated once, for the final model. Choosing between
models on test turns it into a second validation set.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader
from tqdm import tqdm

# make `src` importable; must run before any `from src...` import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.dataset import get_datasets  # noqa: E402
from src.dataset.paths import get_output_dir  # noqa: E402
from src.dataset.transforms import REGIONS  # noqa: E402
from src.evaluation import add_netc, case_metrics, logits_to_masks, predict_logits  # noqa: E402
from src.model import build_unet  # noqa: E402
from src.training.setup import setup_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-name", required=True, help="Training run under outputs/runs/")
    p.add_argument("--checkpoint", default="best", choices=["best", "last"])
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--sw-batch-size", type=int, default=4)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gpu-mem-gb", type=float, default=None)
    p.add_argument("--limit", type=int, default=None, help="Debug: evaluate only the first N cases")
    return p.parse_args()


def summarize(per_case: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        dice, hd = per_case[f"dice_{name}"], per_case[f"hd95_{name}"]
        gt, pred = per_case[f"gt_voxels_{name}"] > 0, per_case[f"pred_voxels_{name}"] > 0
        rows.append({
            "region": name,
            "dice_mean": dice.mean(), "dice_std": dice.std(), "dice_median": dice.median(),
            "hd95_mean": hd.mean(), "hd95_median": hd.median(),
            "cases_with_region": int(gt.sum()),
            "missed_cases": int((gt & ~pred).sum()),            # region present, nothing predicted
            "false_alarm_cases": int((~gt & pred).sum()),       # region absent, something predicted
        })
    return pd.DataFrame(rows).set_index("region")


def main() -> None:
    args = parse_args()
    device, amp = setup_device(args.gpu_mem_gb)

    run_dir = get_output_dir() / "runs" / args.run_name
    ckpt_path = run_dir / "checkpoints" / f"{args.checkpoint}.pt"
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = state["config"]
    include_rc = not config.get("no_rc", False)
    patch = (config["patch"],) * 3
    print(f"Checkpoint: {ckpt_path} (epoch {state['epoch']}, val mean Dice {state['best_dice_mean']:.4f})")

    model = build_unet(include_rc=include_rc).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    dataset = get_datasets(patch_size=patch, include_rc=include_rc)[args.split]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    names = list(REGIONS[: 4 if include_rc else 3]) + ["NETC"]
    print(f"Split: {args.split}, {len(dataset)} cases")

    rows = []
    for i, batch in enumerate(tqdm(loader, desc=args.split)):
        if args.limit is not None and i >= args.limit:
            break
        image = batch["image"].to(device)
        logits = predict_logits(model, image, patch, args.sw_batch_size, args.overlap, amp)
        masks = add_netc(logits_to_masks(logits)[0].cpu())
        label = add_netc(batch["label"][0])
        rows.append({"case_id": batch["case_id"][0], **case_metrics(masks, label, names)})

    out_dir = run_dir / "evaluation" / f"{args.split}_{args.checkpoint}"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_case = pd.DataFrame(rows).set_index("case_id")
    per_case.to_csv(out_dir / "per_case.csv")
    summary = summarize(per_case, names)
    summary.to_csv(out_dir / "summary.csv")

    tumour = [n for n in ("ET", "TC", "WT") if n in names]
    print()
    print(summary.round(4).to_string())
    print(f"\nMean Dice over {', '.join(tumour)}: {np.nanmean(summary.loc[tumour, 'dice_mean']):.4f}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()