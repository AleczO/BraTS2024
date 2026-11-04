"""Per-epoch logging to the console, TensorBoard and metrics.csv."""

import csv
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

FIELDS = ["epoch", "lr", "train_loss", "epoch_time_s", "gpu_mem_gb",
          "dice_ET", "dice_TC", "dice_WT", "dice_RC", "dice_mean", "new_best"]


class RunLogger:
    def __init__(self, run_dir: Path):
        self.writer = SummaryWriter(run_dir / "tensorboard")
        self.metrics_path = run_dir / "metrics.csv"

    def log(self, row: dict) -> None:
        """One epoch: row with 'epoch' and any numeric values (missing fields stay empty)."""
        epoch = row["epoch"]
        for key, value in row.items():
            if key not in ("epoch", "new_best"):
                self.writer.add_scalar(key, value, epoch)

        print(" | ".join(
            f"{k} {v:.2e}" if k == "lr" else f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
            for k, v in row.items()
        ))

        new_file = not self.metrics_path.exists()
        with self.metrics_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new_file:
                w.writeheader()
            w.writerow(row)

    def close(self) -> None:
        self.writer.close()