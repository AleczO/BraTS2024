"""Saving and loading training checkpoints."""

from pathlib import Path

import torch
from torch import nn


def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler,
                    epoch: int, best: float, config: dict) -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,  # last finished epoch
        "best_dice_mean": best,
        "config": config,
    }, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer, scheduler,
                    device: torch.device) -> tuple[int, float]:
    """Restore everything in place; return (epoch to start from, best mean Dice so far)."""
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state["epoch"] + 1, state["best_dice_mean"]