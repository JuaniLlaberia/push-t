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
from .losses import bc_loss
from .data import load_episode_indices, make_dataloaders, load_dataset_stats
from .split import split_episodes
from .utils import plot_losses, run_dir, set_seed, get_device, save_checkpoint, checkpoint_path

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

def train_one_epoch(model, loader, optimizer, normalizer, device, loss_kind: str = "mse", step_losses: list | None = None) -> float:
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
        step_losses: optional list; if given, each step's loss is appended to it.
    Returns:
        Mean training loss over the epoch.
    """
    model.train()

    mean_loss = []
    for batch in loader:
        if "features" in batch:
            features = batch["features"].to(device)
        else:
            features = batch["observation.image"].to(device)
            features = normalizer.normalize_image(features)
        
        state = batch["observation.state"].to(device)
        action = batch["action"].to(device)

        state_normalized = normalizer.normalize_state(state)
        action_normalized = normalizer.normalize_action(action)

        pred = model(features, state_normalized)

        if "action_is_pad" in batch:
            mask = ~batch["action_is_pad"].to(device)
        else:
            mask = None

        loss = bc_loss(pred, action_normalized, kind=loss_kind, mask=mask)
        mean_loss.append(loss.item())
        if step_losses is not None:
            step_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return sum(mean_loss) / len(mean_loss)

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
    model.eval()

    with torch.no_grad():
        mean_loss = []

        for batch in loader:
            if "features" in batch:
                features = batch["features"].to(device)
            else:
                features = batch["observation.image"].to(device)
                features = normalizer.normalize_image(features)
                    
            state = batch["observation.state"].to(device)
            action = batch["action"].to(device)
            
            state_normalized = normalizer.normalize_state(state)
            action_normalized = normalizer.normalize_action(action)
            
            pred = model(features, state_normalized)
            
            if "action_is_pad" in batch:
                mask = ~batch["action_is_pad"].to(device)
            else:
                mask = None
            
            loss = bc_loss(pred, action_normalized, kind=loss_kind, mask=mask)
            mean_loss.append(loss.item())
                
        return sum(mean_loss) / len(mean_loss)

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
    # Set seed
    set_seed(cfg.seed)
    # Get device
    device = get_device()
    # Get episodes indices
    episodes_indices = load_episode_indices(cfg)
    # Prepare data loaders (both for train and validation)
    train_episodes, val_episodes = split_episodes(episodes_indices, cfg.val_fraction, seed=cfg.seed)
    train_loader, val_loader = make_dataloaders(cfg, train_episodes, val_episodes)
    # Prepare normalizer
    dataset_stats = load_dataset_stats(cfg)
    normalizer = build_normalizer(dataset_stats)
    # Load model
    model = build_model(cfg, device)
    # Build optimizer
    optimizer = build_optimizer(cfg, model)

    history = {"train_loss": [], "val_loss": []}
    checkpoint_paths = []

    for epoch in range(1, cfg.epochs + 1):
        # Train and get loss
        epoch_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            normalizer=normalizer,
            device=device,
            loss_kind=cfg.loss,
            step_losses=history["train_loss"]
        )
        # Compute validation loss
        val_epoch_loss = compute_val_loss(
            model=model,
            normalizer=normalizer,
            loader=val_loader,
            device=device,
            loss_kind=cfg.loss
        )
        history["val_loss"].append(val_epoch_loss)

        print(f"epoch {epoch}/{cfg.epochs} | train {epoch_loss:.4f} | val {val_epoch_loss:.4f}")
        
        # Save training checkpoint
        path = save_checkpoint(
            path=checkpoint_path(cfg, epoch),
            model=model,
            normalizer_stats=normalizer.state_dict(),
            config=cfg,
            epoch=epoch
        )
        checkpoint_paths.append(path)

    return TrainResult(
        history=history,
        checkpoint_paths=checkpoint_paths,
        best_val_loss=min(history["val_loss"])
    )

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