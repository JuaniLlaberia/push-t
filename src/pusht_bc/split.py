from __future__ import annotations
import random

def split_episodes(episode_indices: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """
    Split episodes (not frames) into train and validation sets.

    Contract: deterministic for a given seed; disjoint; together cover all inputs;
    len(val) == max(1, round(len(episode_indices) * val_fraction)); both lists sorted, plain ints.

    Args:
        episode_indices: all episode indices.
        val_fraction: fraction of episodes to hold out.
        seed: random seed.
    Returns:
        (train_episodes, val_episodes).
    """
    rd = random.Random(seed)
    episode_indices = sorted([int(e) for e in episode_indices])

    num_val_episodes = max(1, round(len(episode_indices) * val_fraction))
    val_episodes = rd.sample(episode_indices, num_val_episodes)

    train_episodes = set(episode_indices) - set(val_episodes)

    return sorted(train_episodes), sorted(val_episodes)
