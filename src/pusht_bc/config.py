from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repository root (editable install: src/pusht_bc/config.py -> repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class EvalConfig:
    """
    Evaluation settings of an experiment.

    Args:
        dev_seeds: inclusive seed range [first, last] used while developing.
        final_seeds: inclusive seed range [first, last] for the final report.
        max_steps: episode cap.
        record_seeds: seeds recorded as video.
    """

    dev_seeds: list[int] = field(default_factory=lambda: [0, 49])
    final_seeds: list[int] = field(default_factory=lambda: [1000, 1199])
    max_steps: int = 300
    record_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])

    def dev_seed_list(self) -> list[int]:
        """
        Expand dev_seeds into a list.

        Returns:
            Seeds first..last inclusive.
        """
        return _inclusive_range(self.dev_seeds, "eval.dev_seeds")

    def final_seed_list(self) -> list[int]:
        """
        Expand final_seeds into a list.

        Returns:
            Seeds first..last inclusive.
        """
        return _inclusive_range(self.final_seeds, "eval.final_seeds")

@dataclass
class Config:
    """
    Experiment configuration (one YAML file in configs/).

    Args:
        name: run name; outputs go to <output_dir>/<name>/.
        seed: seed for the split, subsampling, shuffling and initialization.
        dataset_repo_id: LeRobot dataset on the Hugging Face Hub.
        val_fraction: fraction of episodes held out for validation.
        train_episode_fraction: fraction of training episodes actually used (E3).
        use_cached_features: train on cached ResNet-18 features instead of images.
        cache_path: feature cache written by scripts/cache_features.py.
        freeze_encoder: keep the ResNet-18 frozen (image path only can set False).
        crop: crop size for the image path (random in training, center at eval), or None.
        hidden_dim: MLP hidden width.
        chunk_size: actions predicted per call (section 10), or None.
        n_execute: actions executed per predicted chunk; required with chunk_size.
        loss: "mse" or "l1".
        batch_size: batch size.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        epochs: training epochs.
        eval: evaluation settings.
        output_dir: root folder for runs.
        num_workers: DataLoader workers for the image path (not in bc_v0.yaml; 0 = main process).
    """

    name: str = "bc_v0"
    seed: int = 0
    dataset_repo_id: str = "lerobot/pusht"
    val_fraction: float = 0.1
    train_episode_fraction: float = 1.0
    use_cached_features: bool = True
    cache_path: str = "data/cache/features_resnet18.pt"
    freeze_encoder: bool = True
    crop: int | None = None
    hidden_dim: int = 256
    chunk_size: int | None = None
    n_execute: int | None = None
    loss: str = "mse"
    batch_size: int = 256
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    epochs: int = 50
    eval: EvalConfig = field(default_factory=EvalConfig)
    output_dir: str = "runs"
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.loss not in ("mse", "l1"):
            raise ValueError(f"loss must be 'mse' or 'l1', got {self.loss!r}")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {self.val_fraction}")
        if not 0.0 < self.train_episode_fraction <= 1.0:
            raise ValueError(f"train_episode_fraction must be in (0, 1], got {self.train_episode_fraction}")
        if self.chunk_size is not None:
            if self.chunk_size < 1:
                raise ValueError("chunk_size must be >= 1 or null")
            if self.n_execute is None or not 1 <= self.n_execute <= self.chunk_size:
                raise ValueError("with chunk_size set, n_execute must be in [1, chunk_size]")
        elif self.n_execute is not None:
            raise ValueError("n_execute requires chunk_size")
        if self.use_cached_features and not self.freeze_encoder:
            raise ValueError("use_cached_features requires freeze_encoder: true")
        if self.use_cached_features and self.crop is not None:
            raise ValueError("crop only applies to the image path (use_cached_features: false)")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to plain Python types.

        Returns:
            Nested dict with the same structure as the YAML file.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        """
        Build a Config from a dict, rejecting unknown keys.

        Args:
            d: dict with the YAML structure.
        Returns:
            The Config.
        """
        d = dict(d)
        _check_keys(d, cls, "config")
        eval_d = d.pop("eval", None) or {}
        _check_keys(eval_d, EvalConfig, "eval")
        return cls(**d, eval=EvalConfig(**eval_d))

def load_config(path: str | Path) -> Config:
    """
    Load a Config from YAML. Unknown keys raise a ValueError.

    Args:
        path: YAML file, e.g. configs/bc_v0.yaml.
    Returns:
        The Config.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return Config.from_dict(raw)

def resolve_path(p: str | Path) -> Path:
    """
    Resolve a config path. Relative paths are relative to the repository root, not the CWD.

    Args:
        p: absolute or repo-relative path.
    Returns:
        Absolute path.
    """
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p

def _check_keys(d: dict, cls: type, where: str) -> None:
    allowed = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - allowed
    if unknown:
        raise ValueError(f"unknown {where} key(s): {sorted(unknown)}; allowed: {sorted(allowed)}")

def _inclusive_range(pair: list[int], name: str) -> list[int]:
    if len(pair) != 2 or pair[0] > pair[1]:
        raise ValueError(f"{name} must be [first, last] with first <= last, got {pair}")
    return list(range(int(pair[0]), int(pair[1]) + 1))
