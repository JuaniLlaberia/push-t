from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import resolve_path
from .data import make_image_transform
from .encoders import make_resnet18_encoder
from .model import BCPolicy
from .normalize import Normalizer
from .utils import get_device, load_checkpoint, plot_coverage_over_time, write_video

SUCCESS_THRESHOLD = 0.95

def make_env(max_steps: int = 300):
    """
    Create gym_pusht/PushT-v0 with pixel + agent-position observations.

    Args:
        max_steps: episode cap (the env truncates there).
    Returns:
        gymnasium env.
    """
    import gymnasium as gym
    import gym_pusht  # noqa: F401  (registers the env)

    return gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=max_steps,
    )

@dataclass
class EpisodeResult:
    """
    Result of one rollout.

    Args:
        seed: env seed.
        success: max coverage >= 0.95.
        max_coverage: maximum coverage over the episode.
        steps: env steps taken.
        rewards: per-step coverage (len == steps).
        frames: uint8 (H, W, 3) images if recorded, else None.
    """

    seed: int
    success: bool
    max_coverage: float
    steps: int
    rewards: list[float]
    frames: list[np.ndarray] | None

@dataclass
class EvalSummary:
    """
    Aggregate over several rollouts.

    Args:
        success_rate: fraction of successful episodes.
        ci_low: lower bound of the 95% Wilson interval.
        ci_high: upper bound of the 95% Wilson interval.
        avg_max_coverage: mean of max_coverage.
        n_episodes: number of episodes.
        episodes: per-episode results, in seed order.
    """

    success_rate: float
    ci_low: float
    ci_high: float
    avg_max_coverage: float
    n_episodes: int
    episodes: list[EpisodeResult]

def binomial_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    Args:
        successes: number of successes.
        n: number of trials (n == 0 gives (0, 1)).
        z: normal quantile (1.96 for 95%).
    Returns:
        (low, high).
    """
    if n <= 0:
        return 0.0, 1.0
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, n], got {successes} / {n}")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)

def load_policy_for_eval(checkpoint_path, device) -> tuple[nn.Module, dict]:
    """
    Rebuild a trained BCPolicy with a frozen ResNet-18 attached and load its weights.

    Args:
        checkpoint_path: file written by save_checkpoint.
        device: target device.
    Returns:
        (model in eval mode on device, normalizer_stats).
    """
    ckpt = load_checkpoint(checkpoint_path)
    cfg = ckpt["config"]
    model = BCPolicy(encoder=make_resnet18_encoder(), hidden_dim=cfg.hidden_dim, chunk_size=cfg.chunk_size)
    model.head.load_state_dict(ckpt["head"])
    if ckpt.get("encoder") is not None:
        model.encoder.load_state_dict(ckpt["encoder"])
    model.to(device)
    model.eval()
    return model, ckpt["normalizer_stats"]

def build_agent_from_checkpoint(checkpoint_path, device=None) -> "BCAgent":
    """
    BCAgent for a checkpoint, with the center crop and n_execute of its config.

    Args:
        checkpoint_path: file written by save_checkpoint.
        device: target device (None = get_device()).
    Returns:
        BCAgent.
    """
    device = torch.device(device) if device is not None else get_device()
    model, stats = load_policy_for_eval(checkpoint_path, device)
    cfg = load_checkpoint(checkpoint_path)["config"]
    return BCAgent(
        model,
        Normalizer.from_stats(stats),
        device,
        image_transform=make_image_transform(cfg.crop, train=False),
        n_execute=cfg.n_execute,
    )

class BCAgent:
    """
    Wraps a trained BCPolicy for closed-loop use (guide sections 7 and 10).

    Same interface as policies.pretrained_dp.PretrainedDiffusionAgent: reset() and act().

    Args:
        model: BCPolicy with its encoder attached (takes (B, 3, H, W) normalized images).
        normalizer: the Normalizer used in training.
        device: device the model lives on.
        image_transform: optional callable on the (3, H, W) [0, 1] image, applied before
            normalization (e.g. the center crop used with cfg.crop). None = identity.
        n_execute: for a chunked model (model.chunk_size = K), how many actions of each
            predicted chunk to execute before predicting again (1 <= n_execute <= K).
            None for a non-chunked model.
    """

    def __init__(self, model, normalizer, device, image_transform=None, n_execute: int | None = None):
        raise NotImplementedError("TODO(core): see guide sections 7 and 10")

    def reset(self) -> None:
        """
        Clear any per-episode state.
        """
        raise NotImplementedError("TODO(core): see guide sections 7 and 10")

    def act(self, obs_batch: dict) -> np.ndarray:
        """
        Choose the next action.

        Args:
            obs_batch: output of env_obs_to_batch (no batch dim).
        Returns:
            (2,) float action in env units, clipped to [0, 512].
        """
        raise NotImplementedError("TODO(core): see guide sections 7 and 10")

def rollout(agent, env, seed: int, max_steps: int = 300, record: bool = False) -> EpisodeResult:
    """
    Run one seeded episode. Success = max coverage >= 0.95 (guide section 7).

    The episode starts from env.reset(seed=seed) with a freshly reset agent and ends when
    the env terminates or truncates, or after max_steps steps, whichever comes first.
    The reward of each step is the current coverage.

    Args:
        agent: object with reset() and act(obs_batch).
        env: gymnasium-like env (make_env()).
        seed: env seed.
        max_steps: step cap.
        record: keep frames.
    Returns:
        EpisodeResult with one reward per step taken (len(rewards) == steps) and frames as
        a list of uint8 (H, W, 3) images if record, else None.
    """
    raise NotImplementedError("TODO(core): see guide section 7")

def evaluate(agent, env, seeds: list[int], record_seeds: tuple[int, ...] = ()) -> EvalSummary:
    """
    Run rollout for each seed; aggregate success rate (+ binomial_ci) and average max coverage (guide section 7).

    Args:
        agent: object with reset() and act(obs_batch).
        env: gymnasium-like env.
        seeds: env seeds, one episode each.
        record_seeds: seeds whose frames are kept.
    Returns:
        EvalSummary with episodes in the order of seeds.
    """
    raise NotImplementedError("TODO(core): see guide section 7")

def parse_seeds(spec: str) -> list[int]:
    """
    Parse a seed list.

    Args:
        spec: '0-49' (inclusive range), '0,1,2', '0-4,10' or ''.
    Returns:
        List of seeds.
    """
    out: list[int] = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                raise ValueError(f"bad seed range {part!r}")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return out

def summary_to_dict(summary: EvalSummary) -> dict:
    """
    JSON-ready version of a summary (frames dropped).

    Args:
        summary: EvalSummary.
    Returns:
        Dict.
    """
    d = asdict(summary)
    for ep in d["episodes"]:
        ep.pop("frames", None)
    return d

def run_and_save(agent, seeds: list[int], record_seeds=(), out_dir=None, max_steps: int = 300, label: str = "") -> EvalSummary:
    """
    Evaluate an agent and write summary.json, seed_<s>.mp4 for recorded seeds and coverage.png.

    Args:
        agent: object with reset() and act(obs_batch).
        seeds: env seeds.
        record_seeds: seeds to record.
        out_dir: output folder (None = write nothing).
        max_steps: episode cap (env truncation).
        label: name stored in summary.json.
    Returns:
        EvalSummary.
    """
    env = make_env(max_steps)
    try:
        summary = evaluate(agent, env, list(seeds), tuple(record_seeds))
    finally:
        env.close()
    if out_dir is not None:
        out = resolve_path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"label": label, "max_steps": max_steps, **summary_to_dict(summary)}, f, indent=2)
        curves = {}
        for ep in summary.episodes:
            if ep.frames:
                write_video(ep.frames, out / f"seed_{ep.seed}.mp4", fps=10)
            if ep.seed in set(record_seeds):
                tag = "success" if ep.success else "fail"
                curves[f"seed {ep.seed} ({tag})"] = ep.rewards
        if curves:
            plot_coverage_over_time(curves, out / "coverage.png")
    return summary

def format_summary(summary: EvalSummary, label: str = "") -> str:
    """
    One-line description of a summary.

    Args:
        summary: EvalSummary.
        label: optional prefix.
    Returns:
        The line.
    """
    head = f"{label}: " if label else ""
    return (
        f"{head}success {summary.success_rate:.1%} "
        f"(95% CI {summary.ci_low:.1%}-{summary.ci_high:.1%}, n={summary.n_episodes}), "
        f"avg max coverage {summary.avg_max_coverage:.3f}"
    )

def main(argv: list[str] | None = None) -> EvalSummary:
    """
    CLI: python -m pusht_bc.evaluate --checkpoint PATH | --pretrained-dp --seeds 0-49 [--record 0,1,2] [--out DIR]

    Args:
        argv: command-line arguments (None = sys.argv).
    Returns:
        EvalSummary.
    """
    ap = argparse.ArgumentParser(description="Evaluate a policy on gym-pusht with fixed seeds.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="checkpoint written by save_checkpoint")
    src.add_argument("--pretrained-dp", action="store_true", help="lerobot/diffusion_pusht")
    ap.add_argument("--seeds", default="0-49", help="e.g. 0-49 or 1000-1199 or 0,1,2")
    ap.add_argument("--record", default="", help="seeds to record as video, e.g. 0,1,2")
    ap.add_argument("--out", default=None, help="output dir (default: next to the checkpoint)")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--device", default=None, help="override device (default: mps > cuda > cpu)")
    args = ap.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    record = parse_seeds(args.record)
    device = torch.device(args.device) if args.device else get_device()

    if args.pretrained_dp:
        from .policies.pretrained_dp import PretrainedDiffusionAgent

        agent = PretrainedDiffusionAgent(device=device)
        label = "diffusion_pusht"
        out = args.out or "runs/diffusion_pusht/eval"
    else:
        ckpt = Path(args.checkpoint)
        agent = build_agent_from_checkpoint(ckpt, device)
        label = f"{ckpt.parent.parent.name}/{ckpt.stem}"
        out = args.out or str(ckpt.parent.parent / "eval" / ckpt.stem)

    print(f"evaluating {label} on {len(seeds)} seeds ({args.seeds}), device {device}")
    summary = run_and_save(agent, seeds, record, out, args.max_steps, label)
    print(format_summary(summary, label))
    print(f"outputs: {resolve_path(out)}")
    return summary

if __name__ == "__main__":
    main()
