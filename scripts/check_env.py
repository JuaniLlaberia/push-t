from __future__ import annotations

import platform

import numpy as np
import torch

from pusht_bc.config import resolve_path
from pusht_bc.evaluate import make_env
from pusht_bc.utils import get_device, write_video

N_STEPS = 50

def main() -> None:
    """
    Install smoke test: one gym-pusht episode with random actions.

    Usage: python scripts/check_env.py
    Prints observation keys, shapes and dtypes and the reward range, and saves
    runs/check_env/random.mp4.
    """
    print(f"python {platform.python_version()} | torch {torch.__version__} | device {get_device()}")
    env = make_env(max_steps=300)
    obs, info = env.reset(seed=0)
    print("observation:")
    for k, v in obs.items():
        v = np.asarray(v)
        print(f"  {k:10s} shape={v.shape} dtype={v.dtype} min={v.min():.2f} max={v.max():.2f}")
    print(f"action space: {env.action_space}")

    frames = [env.render()]
    rewards = []
    for _ in range(N_STEPS):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        rewards.append(float(reward))
        frames.append(env.render())
        if terminated or truncated:
            break
    env.close()

    out = write_video(frames, resolve_path("runs/check_env/random.mp4"), fps=10)
    print(f"steps: {len(rewards)} | reward min={min(rewards):.3f} max={max(rewards):.3f}")
    print(f"render frame: {frames[0].shape} {frames[0].dtype}")
    print(f"video: {out}")

if __name__ == "__main__":
    main()
