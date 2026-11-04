"""Train the 3D U-Net on preprocessed BraTS 2024 GLI data.

Run from the repo root (after scripts/preprocess.py):
    python scripts/train.py --run-name unet_baseline
    python scripts/train.py --run-name unet_baseline --resume      # continue from last.pt

Everything for one run goes to outputs/runs/<run-name>/:
    config.json          all settings + git commit, for reproducibility
    metrics.csv          one row per epoch (loss, lr, validation Dice per region)
    tensorboard/         the same curves for TensorBoard
    checkpoints/last.pt  latest state (model, optimizer, scheduler, epoch)
    checkpoints/best.pt  model with the best mean validation Dice
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from monai.utils import set_determinism
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.dataset import get_dataloaders  # noqa: E402
from src.evaluation import evaluate_dice  # noqa: E402
from src.losses import DiceBCELoss  # noqa: E402
from src.model import build_unet, count_parameters  # noqa: E402
from src.training.checkpoints import load_checkpoint, save_checkpoint  # noqa: E402
from src.training.logger import RunLogger  # noqa: E402
from src.training.setup import create_run_dir, save_config, setup_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-name", required=True, help="Folder name under outputs/runs/")
    p.add_argument("--resume", action="store_true", help="Continue from checkpoints/last.pt of this run")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2, help="Cases per batch")
    p.add_argument("--num-samples", type=int, default=2, help="Patches per case (batch = batch_size * num_samples)")
    p.add_argument("--patch", type=int, default=128, help="Cubic patch size, divisible by 16")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--no-rc", action="store_true", help="Train only ET, TC, WT (no resection cavity channel)")

    p.add_argument("--val-every", type=int, default=5, help="Validate every N epochs")
    p.add_argument("--sw-batch-size", type=int, default=4, help="Windows per forward pass in validation")
    p.add_argument("--overlap", type=float, default=0.5, help="Sliding-window overlap in validation")

    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="Disable bfloat16 mixed precision")
    p.add_argument("--gpu-mem-gb", type=float, default=None,
                   help="Cap this process's GPU memory in GB (shared GPU), e.g. 13")
    p.add_argument("--limit-batches", type=int, default=None,
                   help="Debug: stop each epoch after N batches and validate on N cases")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = vars(args)
    set_determinism(seed=args.seed)

    device, amp = setup_device(args.gpu_mem_gb, use_amp=not args.no_amp)
    patch = (args.patch,) * 3
    include_rc = not args.no_rc

    run_dir = create_run_dir(args.run_name, args.resume)
    ckpt_dir = run_dir / "checkpoints"
    if not args.resume:
        save_config(run_dir, config, device)

    # ----- data, model, loss, optimizer -----
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.workers,
        patch_size=patch, num_samples=args.num_samples, include_rc=include_rc,
    )

    model = build_unet(include_rc=include_rc).to(device)
    loss_fn = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch, best = 0, -1.0
    if args.resume:
        start_epoch, best = load_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, device)
        print(f"Resumed from epoch {start_epoch - 1}, best mean Dice {best:.4f}")

    print(f"Run: {run_dir}")
    print(f"Model: {count_parameters(model) / 1e6:.1f} M parameters | device {device} | amp {amp}")
    print(f"Train: {len(train_loader.dataset)} cases, {len(train_loader)} batches/epoch "
          f"of {args.batch_size}×{args.num_samples} patches | Val: {len(val_loader.dataset)} cases")

    logger = RunLogger(run_dir)

    # ----- training loop -----
    for epoch in range(start_epoch, args.epochs):
        model.train()
        torch.cuda.reset_peak_memory_stats(device)

        t0 = time.time()
        running, n_batches = 0.0, 0

        for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(image)
            loss = loss_fn(logits.float(), label)  # loss in float32 for stability
            loss.backward()
            optimizer.step()

            running += loss.item()
            n_batches += 1
            if args.limit_batches and n_batches >= args.limit_batches:
                break

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        row = {"epoch": epoch, "lr": lr, "train_loss": running / n_batches,
               "epoch_time_s": round(time.time() - t0, 1)}
               
        row["gpu_mem_gb"] = round(torch.cuda.max_memory_reserved(device) / 1e9, 2)

        # ----- validation -----
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            scores = evaluate_dice(model, val_loader, device, roi_size=patch,
                                   sw_batch_size=args.sw_batch_size, overlap=args.overlap, amp=amp,
                                   max_cases=args.limit_batches)
            row.update(scores)
            if scores["dice_mean"] > best:
                best = scores["dice_mean"]
                save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, epoch, best, config)
                row["new_best"] = True

        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, epoch, best, config)
        logger.log(row)

    logger.close()
    print(f"Done. Best mean validation Dice: {best:.4f} ({ckpt_dir / 'best.pt'})")


if __name__ == "__main__":
    main()