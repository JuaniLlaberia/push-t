from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .config import Config, load_config
from .encoders import make_resnet18_encoder
from .model import BCPolicy
from .normalize import Normalizer
from .utils import plot_losses, run_dir

@dataclass
class TrainResult:
    """
    Output of train().

    Args:
        history: {"train_loss": [float per optimizer step], "val_loss": [float per epoch]}.
        checkpoint_paths: one per epoch, in epoch order (epochs numbered from 1), written to
            utils.checkpoint_path(cfg, epoch).
        best_val_loss: min(history["val_loss"]).
    """

    history: dict
    checkpoint_paths: list[str]
    best_val_loss: float

def build_model(cfg: Config, device: torch.device) -> BCPolicy:
    """
    Construct the policy (no training logic).

    Args:
        cfg: experiment config. Cached path: encoder=None. Image path: ResNet-18,
            frozen per cfg.freeze_encoder.
        device: target device.
    Returns:
        BCPolicy on device.
    """
    encoder = None
    if not cfg.use_cached_features:
        encoder = make_resnet18_encoder(pretrained=True, frozen=cfg.freeze_encoder)
    model = BCPolicy(encoder=encoder, hidden_dim=cfg.hidden_dim, chunk_size=cfg.chunk_size)
    return model.to(device)

def build_optimizer(cfg: Config, model: nn.Module) -> torch.optim.Optimizer:
    """
    AdamW over the parameters with requires_grad=True.

    Args:
        cfg: experiment config (lr, weight_decay).
        model: the policy.
    Returns:
        The optimizer.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

def build_normalizer(dataset_stats: dict) -> Normalizer:
    """
    Construct the Normalizer.

    Args:
        dataset_stats: LeRobot-style stats (data.load_dataset_stats(cfg)).
    Returns:
        Normalizer.from_stats(dataset_stats).
    """
    return Normalizer.from_stats(dataset_stats)

def train_one_epoch(model, loader, optimizer, normalizer, device, loss_kind: str = "mse") -> float:
    """
    One pass over loader (guide section 6).

    Args:
        model: BCPolicy.
        loader: yields raw dicts: "features" (B, 512) or "observation.image" (B, 3, H, W)
            in [0, 1], "observation.state" (B, 2), "action" (B, 2) or (B, K, 2), and
            optionally "action_is_pad" (B, K) bool.
        optimizer: optimizer over the trainable parameters.
        normalizer: Normalizer.
        device: device the model lives on.
        loss_kind: "mse" or "l1" (cfg.loss), passed through to bc_loss.
    Returns:
        Mean training loss over the epoch.
    """
    raise NotImplementedError("TODO(core): see guide section 6")

def compute_val_loss(model, loader, normalizer, device, loss_kind: str = "mse") -> float:
    """
    Mean loss over loader without updating parameters (guide section 6).

    Args:
        model: BCPolicy.
        loader: same batch format as in train_one_epoch.
        normalizer: Normalizer.
        device: device the model lives on.
        loss_kind: "mse" or "l1".
    Returns:
        Mean loss as a Python float.
    """
    raise NotImplementedError("TODO(core): see guide section 6")

def train(cfg: Config) -> TrainResult:
    """
    Build data/model/optimizer via the provided factories, run cfg.epochs, log history
    (train loss per step, val loss per epoch), save a checkpoint every epoch (guide section 6).

    Provided plumbing: data.load_episode_indices(cfg), data.load_dataset_stats(cfg),
    data.make_dataloaders(cfg, train_eps, val_eps), split.split_episodes,
    utils.set_seed, utils.get_device, utils.save_checkpoint, utils.checkpoint_path,
    and build_model, build_optimizer, build_normalizer in this module.

    Args:
        cfg: experiment config.
    Returns:
        TrainResult.
    """
    raise NotImplementedError("TODO(core): see guide section 6")

def save_train_outputs(cfg: Config, result: TrainResult) -> Path:
    """
    Write <run>/history.json and <run>/losses.png.

    Args:
        cfg: experiment config.
        result: output of train(cfg).
    Returns:
        The run directory.
    """
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "history.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "history": result.history,
                "checkpoint_paths": [str(p) for p in result.checkpoint_paths],
                "best_val_loss": float(result.best_val_loss),
                "config": cfg.to_dict(),
            },
            f,
            indent=2,
        )
    plot_losses(result.history, out / "losses.png")
    return out

def main(argv: list[str] | None = None) -> TrainResult:
    """
    CLI: python -m pusht_bc.train --config configs/bc_v0.yaml [--name NAME]

    Args:
        argv: command-line arguments (None = sys.argv).
    Returns:
        TrainResult.
    """
    ap = argparse.ArgumentParser(description="Train a BC policy on Push-T.")
    ap.add_argument("--config", required=True, help="YAML config, e.g. configs/bc_v0.yaml")
    ap.add_argument("--name", default=None, help="override cfg.name (output folder under runs/)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.name:
        cfg.name = args.name
    result = train(cfg)
    out = save_train_outputs(cfg, result)
    last = result.checkpoint_paths[-1] if result.checkpoint_paths else "-"
    print(f"best val loss: {result.best_val_loss:.6f}")
    print(f"checkpoints: {len(result.checkpoint_paths)} (last: {last})")
    print(f"outputs: {out}")
    return result

if __name__ == "__main__":
    main()
