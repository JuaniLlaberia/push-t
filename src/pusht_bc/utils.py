from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import Config, resolve_path

def set_seed(seed: int) -> None:
    """
    Seed Python, NumPy and torch (CPU, CUDA and MPS when present).

    Args:
        seed: the seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def get_device() -> torch.device:
    """
    Pick the best available device: mps, then cuda, then cpu.

    Returns:
        The torch device.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def run_dir(cfg: Config) -> Path:
    """
    Output folder of a run.

    Args:
        cfg: experiment config.
    Returns:
        <output_dir>/<name>/, resolved against the repo root.
    """
    return resolve_path(cfg.output_dir) / cfg.name

def checkpoint_path(cfg: Config, epoch: int) -> Path:
    """
    Where the checkpoint of an epoch goes.

    Args:
        cfg: experiment config.
        epoch: epoch number (from 1).
    Returns:
        <output_dir>/<name>/checkpoints/epoch_XXX.pt
    """
    return run_dir(cfg) / "checkpoints" / f"epoch_{epoch:03d}.pt"

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    normalizer_stats: dict,
    config: Config,
    epoch: int,
    extra: dict | None = None,
) -> Path:
    """
    Save head weights, encoder weights only if the encoder was trained, and metadata.

    A frozen pretrained encoder is never saved; it is rebuilt from torchvision at load time.

    Args:
        path: output .pt file (parent folders are created).
        model: follows the BCPolicy attribute convention, model.head (nn.Module) and
            model.encoder (nn.Module or None).
        normalizer_stats: Normalizer.state_dict().
        config: experiment config; config.freeze_encoder decides whether the encoder is saved.
        epoch: epoch number.
        extra: optional extra metadata (e.g. losses).
    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder = getattr(model, "encoder", None)
    save_encoder = encoder is not None and not config.freeze_encoder
    payload = {
        "head": {k: v.detach().cpu() for k, v in model.head.state_dict().items()},
        "encoder": (
            {k: v.detach().cpu() for k, v in encoder.state_dict().items()} if save_encoder else None
        ),
        "normalizer_stats": _to_cpu(normalizer_stats),
        "config": config.to_dict(),
        "epoch": int(epoch),
        "extra": _to_cpu(extra or {}),
    }
    torch.save(payload, path)
    return path

def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """
    Load a checkpoint written by save_checkpoint. Tensors are on CPU.

    Args:
        path: .pt file.
    Returns:
        Dict with "head", "encoder" (state dict or None), "normalizer_stats",
        "config" (a Config), "epoch" and "extra".
    """
    # Our own files; they hold numpy arrays and plain dicts, so weights_only=False.
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    ckpt["config"] = Config.from_dict(ckpt["config"])
    return ckpt

def _to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj

def write_video(frames: list[np.ndarray], path: str | Path, fps: int = 10) -> Path:
    """
    Write frames to an H.264 mp4. Odd sizes are padded by one pixel.

    Args:
        frames: uint8 (H, W, 3) images.
        path: output .mp4 file (parent folders are created).
        fps: frames per second.
    Returns:
        The path written.
    """
    import imageio.v2 as imageio

    if len(frames) == 0:
        raise ValueError("write_video: no frames")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1, quality=8) as w:
        for f in frames:
            f = np.asarray(f)
            if f.dtype != np.uint8:
                f = np.clip(f, 0, 255).astype(np.uint8)
            h, w_ = f.shape[:2]
            if h % 2 or w_ % 2:
                f = np.pad(f, ((0, h % 2), (0, w_ % 2), (0, 0)), mode="edge")
            w.append_data(f)
    return path

def plot_losses(history: dict, path: str | Path) -> Path:
    """
    Plot train loss per step and val loss per epoch (drawn where each epoch ends).

    Args:
        history: {"train_loss": [per step], "val_loss": [per epoch]}.
        path: output .png file.
    Returns:
        The path written.
    """
    from matplotlib.figure import Figure  # backend-free: works headless and in notebooks

    train = list(history.get("train_loss", []))
    val = list(history.get("val_loss", []))
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    if train:
        ax.plot(np.arange(1, len(train) + 1), train, lw=0.8, alpha=0.8, label="train (per step)")
    if val:
        steps_per_epoch = len(train) / len(val) if train else 1.0
        xs = steps_per_epoch * np.arange(1, len(val) + 1)
        ax.plot(xs, val, "o-", ms=3, label="val (per epoch)")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (normalized units)")
    if any(v > 0 for v in train + val):
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    return path

def plot_coverage_over_time(rewards_by_label: dict[str, list[float]], path: str | Path) -> Path:
    """
    Plot coverage curves on one axis, with the 0.95 success line.

    Args:
        rewards_by_label: label -> per-step coverage.
        path: output .png file.
    Returns:
        The path written.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    for label, rewards in rewards_by_label.items():
        ax.plot(np.arange(1, len(rewards) + 1), rewards, label=label)
    ax.axhline(0.95, color="k", ls="--", lw=1, label="success (0.95)")
    ax.set_xlabel("step")
    ax.set_ylabel("coverage")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    return path
