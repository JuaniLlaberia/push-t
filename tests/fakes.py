from __future__ import annotations

import numpy as np
import torch
from torch import nn

class FakePushTEnv:
    """
    Gymnasium-like env that plays back scripted coverage values.

    Like gym-pusht 0.1.8, step() returns reward = clip(coverage / 0.95, 0, 1) and puts the
    coverage itself in info["coverage"], so code that treats the reward as coverage fails.

    After the script runs out, the last reward repeats. Observations match gym-pusht:
    "pixels" uint8 (96, 96, 3), "agent_pos" float64 (2,). Every action received is stored
    in self.actions (current episode) and self.all_actions; reset seeds in self.reset_seeds.

    Args:
        rewards: per-step coverage values, or {seed: list} for seed-dependent episodes.
        terminate_at_success: terminated=True when coverage >= 0.95.
        truncate_after: truncated=True at this step count (None = never).
    """

    def __init__(self, rewards, terminate_at_success: bool = True, truncate_after: int | None = None):
        self._script = rewards
        self.terminate_at_success = terminate_at_success
        self.truncate_after = truncate_after
        self.actions: list[np.ndarray] = []
        self.all_actions: list[np.ndarray] = []
        self.reset_seeds: list[int | None] = []
        self.t = 0
        self._rewards: list[float] = []
        self._rng = np.random.default_rng(0)

    def _obs(self) -> dict:
        return {
            "pixels": self._rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8),
            "agent_pos": self._rng.uniform(20.0, 490.0, size=2).astype(np.float64),
        }

    def reset(self, seed: int | None = None, options=None):
        self.reset_seeds.append(seed)
        self._rng = np.random.default_rng(seed)
        script = self._script[seed] if isinstance(self._script, dict) else self._script
        self._rewards = [float(r) for r in script]
        if not self._rewards:
            raise ValueError("empty reward script")
        self.t = 0
        self.actions = []
        return self._obs(), {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).copy()
        self.actions.append(a)
        self.all_actions.append(a)
        coverage = self._rewards[min(self.t, len(self._rewards) - 1)]
        reward = float(np.clip(coverage / 0.95, 0.0, 1.0))
        self.t += 1
        terminated = bool(self.terminate_at_success and coverage >= 0.95)
        truncated = bool(self.truncate_after is not None and self.t >= self.truncate_after)
        info = {"coverage": coverage, "is_success": coverage >= 0.95}
        return self._obs(), reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return np.full((96, 96, 3), min(self.t, 255), dtype=np.uint8)

    def close(self) -> None:
        pass

class ConstantModel(nn.Module):
    """
    BCPolicy call signature; returns a fixed normalized output for every batch element.

    self.calls counts forward calls; self.last_inputs keeps the last (obs, state) received.

    Args:
        output: (2,) -> forward returns (B, 2); (K, 2) -> forward returns (B, K, 2).
    """

    def __init__(self, output):
        super().__init__()
        output = torch.as_tensor(output, dtype=torch.float32)
        if output.ndim not in (1, 2) or output.shape[-1] != 2:
            raise ValueError("output must be (2,) or (K, 2)")
        self.register_buffer("output", output.clone())
        self.dummy = nn.Parameter(torch.zeros(1))
        self.encoder = None
        self.head = nn.Identity()
        self.chunk_size = output.shape[0] if output.ndim == 2 else None
        self.calls = 0
        self.last_inputs: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(self, obs_feat_or_image: torch.Tensor, state_norm: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.last_inputs = (obs_feat_or_image.detach().clone(), state_norm.detach().clone())
        b = obs_feat_or_image.shape[0]
        out = self.output.to(obs_feat_or_image.device)
        return out.expand(b, *out.shape).clone() + 0.0 * self.dummy

class FixedAgent:
    """
    Agent interface with a constant action; counts reset() and act() calls.

    Args:
        action: the action returned by every act().
    """

    def __init__(self, action=(256.0, 256.0)):
        self.action = np.asarray(action, dtype=np.float64)
        self.resets = 0
        self.acts = 0

    def reset(self) -> None:
        self.resets += 1

    def act(self, obs_batch: dict) -> np.ndarray:
        self.acts += 1
        return self.action.copy()

STATE_MIN = np.array([10.0, 20.0])
STATE_MAX = np.array([490.0, 500.0])
ACTION_MIN = np.array([12.0, 25.0])
ACTION_MAX = np.array([511.0, 505.0])
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def synthetic_stats() -> dict:
    """
    LeRobot-style stats with known min/max.

    Returns:
        Stats dict (numpy float64, as in dataset.meta.stats).
    """
    def entry(lo, hi):
        return {
            "min": lo.copy(),
            "max": hi.copy(),
            "mean": (lo + hi) / 2,
            "std": (hi - lo) / 4,
            "count": np.array([1000]),
        }

    return {
        "observation.state": entry(STATE_MIN, STATE_MAX),
        "action": entry(ACTION_MIN, ACTION_MAX),
        "observation.image": {
            "min": np.zeros((3, 1, 1)),
            "max": np.ones((3, 1, 1)),
            "mean": IMAGENET_MEAN.reshape(3, 1, 1).copy(),
            "std": IMAGENET_STD.reshape(3, 1, 1).copy(),
            "count": np.array([1000]),
        },
    }

def make_synthetic_cache(episode_lengths: list[int], feat_dim: int = 512, seed: int = 0) -> dict:
    """
    Tiny cache in the format written by scripts/cache_features.py.

    action[row] = (1000 * episode + frame, -(1000 * episode + frame)), so tests can tell
    which episode and frame an action came from.

    Args:
        episode_lengths: number of frames per episode.
        feat_dim: feature size.
        seed: seed for features and states.
    Returns:
        Cache dict.
    """
    g = torch.Generator().manual_seed(seed)
    ep, fr = [], []
    for e, n in enumerate(episode_lengths):
        ep += [e] * n
        fr += list(range(n))
    ep_t = torch.tensor(ep, dtype=torch.long)
    fr_t = torch.tensor(fr, dtype=torch.long)
    code = (1000 * ep_t + fr_t).float()
    n = len(ep)
    return {
        "features": torch.randn(n, feat_dim, generator=g),
        "state": torch.rand(n, 2, generator=g) * 400 + 50,
        "action": torch.stack([code, -code], dim=1),
        "episode_index": ep_t,
        "frame_index": fr_t,
        "index": torch.arange(n),
        "stats": synthetic_stats(),
    }
