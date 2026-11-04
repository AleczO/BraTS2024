"""Run setup: device, run folder, saved configuration."""

import json
import subprocess
import sys
from pathlib import Path

import torch

from src.dataset.paths import REPO_ROOT, get_output_dir


def setup_device(gpu_mem_gb: float | None = None, use_amp: bool = True) -> tuple[torch.device, bool]:
    """Use the GPU (stop if none is visible); optionally cap this process's GPU memory in GB."""
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU visible to PyTorch. Check `nvidia-smi` and the installed torch build "
                 "(python -c \"import torch; print(torch.__version__, torch.version.cuda)\").")
    device = torch.device("cuda", 0)  # explicit index, required by the memory functions
    if gpu_mem_gb:
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        fraction = min(gpu_mem_gb / total, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        print(f"GPU memory capped at {gpu_mem_gb:.1f} GB of {total:.1f} GB ({fraction:.0%})")
    return device, use_amp


def git_commit() -> str:
    """Hash of the current commit, so a run can be traced back to the exact code."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def create_run_dir(run_name: str, resume: bool) -> Path:
    """outputs/runs/<run_name>/ with a checkpoints/ subfolder; refuse to overwrite a run."""
    run_dir = get_output_dir() / "runs" / run_name
    if run_dir.exists() and not resume:
        sys.exit(f"{run_dir} already exists. Use another --run-name or add --resume.")
    if resume and not run_dir.exists():
        sys.exit(f"{run_dir} does not exist, nothing to resume.")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir: Path, config: dict, device: torch.device) -> None:
    """All settings of the run plus the git commit and device."""
    full = {**config, "git_commit": git_commit(), "device": str(device)}
    (run_dir / "config.json").write_text(json.dumps(full, indent=2))