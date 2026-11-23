from __future__ import annotations

def split_episodes(episode_indices: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """
    Split episodes into train and validation sets (guide section 2).

    Contract: deterministic for a given seed; disjoint; together cover all inputs;
    len(val) == max(1, round(len(episode_indices) * val_fraction)); both lists sorted, plain ints.

    Args:
        episode_indices: all episode indices.
        val_fraction: fraction of episodes to hold out.
        seed: random seed.
    Returns:
        (train_episodes, val_episodes).
    """
    raise NotImplementedError("TODO(core): see guide section 2")
