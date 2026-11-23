from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .config import Config, resolve_path

# Everything in this module returns RAW (unnormalized) values; the Normalizer is applied
# by the training and evaluation code.
IMAGE_KEY = "observation.image"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
PAD_KEY = "action_is_pad"
FEATURES_KEY = "features"

def load_dataset(repo_id: str = "lerobot/pusht", delta_timestamps: dict | None = None, image_transforms=None):
    """
    Load a LeRobotDataset (downloaded to the Hugging Face cache on first use).

    Args:
        repo_id: dataset on the Hub.
        delta_timestamps: e.g. {"action": [0.0, 0.1, ...]} for action chunks.
        image_transforms: callable applied to each (3, H, W) image.
    Returns:
        LeRobotDataset. Samples are dicts; "observation.image" is float32 (3, 96, 96) in
        [0, 1], "observation.state" and "action" are float32 (2,) in pixels (0..512).
    """
    from lerobot.datasets import LeRobotDataset

    return LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, image_transforms=image_transforms)

def load_metadata(repo_id: str = "lerobot/pusht"):
    """
    Load dataset metadata only (episodes, stats, fps). Fast; does not download videos.

    Args:
        repo_id: dataset on the Hub.
    Returns:
        LeRobotDatasetMetadata.
    """
    from lerobot.datasets import LeRobotDatasetMetadata

    return LeRobotDatasetMetadata(repo_id)

def _meta(dataset_or_meta):
    return getattr(dataset_or_meta, "meta", dataset_or_meta)

def get_episode_indices(dataset) -> list[int]:
    """
    List the episodes of a dataset.

    Args:
        dataset: LeRobotDataset or LeRobotDatasetMetadata.
    Returns:
        Sorted episode indices.
    """
    eps = _meta(dataset).episodes
    return sorted(int(e) for e in eps["episode_index"])

def frames_for_episodes(dataset, episodes: list[int]) -> list[int]:
    """
    Frame (dataset) indices belonging to some episodes.

    Args:
        dataset: LeRobotDataset or LeRobotDatasetMetadata of the full dataset.
        episodes: episode indices.
    Returns:
        Frame indices in dataset order.
    """
    eps = _meta(dataset).episodes
    wanted = {int(e) for e in episodes}
    known = {int(e) for e in eps["episode_index"]}
    missing = wanted - known
    if missing:
        raise ValueError(f"unknown episode indices: {sorted(missing)[:10]}")
    out: list[int] = []
    for ep, start, stop in zip(eps["episode_index"], eps["dataset_from_index"], eps["dataset_to_index"]):
        if int(ep) in wanted:
            out.extend(range(int(start), int(stop)))
    return sorted(out)

def get_dataset_stats(repo_id: str = "lerobot/pusht") -> dict:
    """
    Stats of the whole dataset.

    Args:
        repo_id: dataset on the Hub.
    Returns:
        LeRobot-style stats: stats[key]["min"|"max"|"mean"|"std"] as numpy float64 arrays.
    """
    return load_metadata(repo_id).stats

def env_obs_to_batch(obs: dict) -> dict[str, torch.Tensor]:
    """
    Convert an env observation to the format of one dataset sample (no batch dim, no normalization).

    Args:
        obs: env observation with "pixels" uint8 (96, 96, 3) and "agent_pos" float (2,).
    Returns:
        {"observation.image": float32 (3, 96, 96) in [0, 1], "observation.state": float32 (2,)}.
    """
    pixels = np.asarray(obs["pixels"])
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise ValueError(f"expected uint8 (H, W, 3) pixels, got {pixels.dtype} {pixels.shape}")
    image = torch.from_numpy(pixels.copy()).permute(2, 0, 1).contiguous().to(torch.float32) / 255.0
    state = torch.as_tensor(np.asarray(obs["agent_pos"]), dtype=torch.float32).reshape(2)
    return {IMAGE_KEY: image, STATE_KEY: state}

def make_image_transform(crop: int | None, train: bool):
    """
    Crop transform on float (..., 3, H, W) tensors, for dataset samples and env_obs_to_batch output.

    Args:
        crop: crop size, or None for no transform.
        train: random crop if True, center crop if False.
    Returns:
        The transform, or None if crop is None.
    """
    if crop is None:
        return None
    from torchvision.transforms import v2

    return v2.RandomCrop(crop) if train else v2.CenterCrop(crop)

@lru_cache(maxsize=2)
def _load_cache_cached(path: str, mtime: float) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)

def load_cache(cache_path: str | Path) -> dict:
    """
    Load the feature cache (keys documented in scripts/cache_features.py).

    Args:
        cache_path: absolute or repo-relative path.
    Returns:
        The cache dict.
    """
    p = resolve_path(cache_path)
    if not p.exists():
        raise FileNotFoundError(
            f"feature cache not found: {p}. Run `python scripts/cache_features.py` first "
            "(needs the Normalizer from guide section 3)."
        )
    return _load_cache_cached(str(p), p.stat().st_mtime)

def cache_rows_for_episodes(cache: dict, episodes: list[int]) -> list[int]:
    """
    Cache rows belonging to some episodes.

    Args:
        cache: loaded cache dict.
        episodes: episode indices.
    Returns:
        Row indices in cache order.
    """
    ep = cache["episode_index"]
    mask = torch.isin(ep, torch.as_tensor(sorted(int(e) for e in episodes), dtype=ep.dtype))
    return torch.nonzero(mask).flatten().tolist()

class CachedFeatureDataset(Dataset):
    """
    Precomputed ResNet-18 features with raw state and action.

    Items: {"features": (512,), "observation.state": (2,), "action": (2,)}. With
    chunk_size=K, "action" is (K, 2), the actions at t, t+1, ..., t+K-1 of the same
    episode, plus "action_is_pad" (K,) bool. Steps past the episode end repeat the
    episode's last action and are marked True, as with LeRobot's delta_timestamps.

    Args:
        cache_path: path to the .pt cache, or an already-loaded cache dict.
        frame_indices: cache rows to expose (row i = dataset frame i).
        chunk_size: K, or None for single actions.
    """

    def __init__(self, cache_path, frame_indices: list[int], chunk_size: int | None = None):
        cache = cache_path if isinstance(cache_path, dict) else load_cache(cache_path)
        self.features = cache["features"].float()
        self.state = cache["state"].float()
        self.action = cache["action"].float()
        self.episode_index = cache["episode_index"].long()
        self.frame_index = cache["frame_index"].long()
        self.stats = cache.get("stats")
        self.frame_indices = [int(i) for i in frame_indices]
        self.chunk_size = chunk_size
        n = len(self.features)
        if not (len(self.state) == len(self.action) == len(self.episode_index) == len(self.frame_index) == n):
            raise ValueError("cache tensors have inconsistent lengths")
        if self.frame_indices and (min(self.frame_indices) < 0 or max(self.frame_indices) >= n):
            raise IndexError("frame index out of range for the cache")
        self._episode_last_row = self._compute_episode_last_row()

    def _compute_episode_last_row(self) -> torch.Tensor:
        ep = self.episode_index
        n = len(ep)
        last = torch.empty(n, dtype=torch.long)
        boundaries = torch.nonzero(ep[1:] != ep[:-1]).flatten() + 1
        starts = torch.cat([torch.tensor([0]), boundaries])
        stops = torch.cat([boundaries, torch.tensor([n])])
        seen = set()
        for s, e in zip(starts.tolist(), stops.tolist()):
            e_id = int(ep[s])
            if e_id in seen:
                raise ValueError(f"episode {e_id} is not contiguous in the cache")
            seen.add(e_id)
            if not torch.all(self.frame_index[s + 1 : e] > self.frame_index[s : e - 1]):
                raise ValueError(f"episode {e_id} frames are not in order in the cache")
            last[s:e] = e - 1
        return last

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        row = self.frame_indices[i]
        item = {
            FEATURES_KEY: self.features[row],
            STATE_KEY: self.state[row],
        }
        if self.chunk_size is None:
            item[ACTION_KEY] = self.action[row]
        else:
            want = row + torch.arange(self.chunk_size)
            last = self._episode_last_row[row]
            item[ACTION_KEY] = self.action[torch.clamp(want, max=last)]
            item[PAD_KEY] = want > last
        return item

class _SelectKeys(Dataset):
    # Keeps only the keys the training loop uses (drops strings like "task").
    def __init__(self, base: Dataset, keys: tuple[str, ...]):
        self.base = base
        self.keys = keys

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = self.base[i]
        return {k: item[k] for k in self.keys if k in item}

def load_episode_indices(cfg: Config) -> list[int]:
    """
    All episode indices of the training data source.

    Args:
        cfg: experiment config (cache if cfg.use_cached_features, else LeRobot metadata).
    Returns:
        Sorted episode indices.
    """
    if cfg.use_cached_features:
        return sorted(set(load_cache(cfg.cache_path)["episode_index"].tolist()))
    return get_episode_indices(load_metadata(cfg.dataset_repo_id))

def load_dataset_stats(cfg: Config) -> dict:
    """
    Stats of the whole dataset.

    Args:
        cfg: experiment config (cache if cfg.use_cached_features, else LeRobot metadata).
    Returns:
        LeRobot-style stats dict.
    """
    if cfg.use_cached_features:
        stats = load_cache(cfg.cache_path).get("stats")
        if stats is not None:
            return stats
    return get_dataset_stats(cfg.dataset_repo_id)

def make_dataloaders(cfg: Config, train_episodes: list[int], val_episodes: list[int]) -> tuple[DataLoader, DataLoader]:
    """
    Build train and val loaders yielding RAW batches.

    Cached path: "features" (B, 512), "observation.state" (B, 2), "action" (B, 2), or
    (B, K, 2) plus "action_is_pad" (B, K) with cfg.chunk_size. Image path:
    "observation.image" (B, 3, H, W) in [0, 1] instead of "features" (H = W = cfg.crop if
    set: random crop for train, center crop for val). Training episodes are subsampled
    to cfg.train_episode_fraction with cfg.seed.

    Args:
        cfg: experiment config.
        train_episodes: training episode indices.
        val_episodes: validation episode indices (must not overlap).
    Returns:
        (train_loader, val_loader).
    """
    train_episodes = sorted(int(e) for e in train_episodes)
    if cfg.train_episode_fraction < 1.0:
        k = max(1, round(len(train_episodes) * cfg.train_episode_fraction))
        train_episodes = sorted(random.Random(cfg.seed).sample(train_episodes, k))
    overlap = set(train_episodes) & set(val_episodes)
    if overlap:
        raise ValueError(f"train and val episodes overlap: {sorted(overlap)[:10]}")

    if cfg.use_cached_features:
        cache = load_cache(cfg.cache_path)
        train_ds = CachedFeatureDataset(cache, cache_rows_for_episodes(cache, train_episodes), cfg.chunk_size)
        val_ds = CachedFeatureDataset(cache, cache_rows_for_episodes(cache, val_episodes), cfg.chunk_size)
        num_workers = 0
    else:
        delta = None
        if cfg.chunk_size is not None:
            fps = load_metadata(cfg.dataset_repo_id).fps
            delta = {ACTION_KEY: [i / fps for i in range(cfg.chunk_size)]}
        keys = (IMAGE_KEY, STATE_KEY, ACTION_KEY) + ((PAD_KEY,) if cfg.chunk_size is not None else ())
        train_base = load_dataset(cfg.dataset_repo_id, delta, make_image_transform(cfg.crop, train=True))
        val_base = load_dataset(cfg.dataset_repo_id, delta, make_image_transform(cfg.crop, train=False))
        train_ds = _SelectKeys(Subset(train_base, frames_for_episodes(train_base, train_episodes)), keys)
        val_ds = _SelectKeys(Subset(val_base, frames_for_episodes(val_base, val_episodes)), keys)
        num_workers = cfg.num_workers

    pin = torch.cuda.is_available()
    g = torch.Generator().manual_seed(cfg.seed)
    common = dict(num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=g, drop_last=False, **common)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader
