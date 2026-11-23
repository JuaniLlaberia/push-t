from __future__ import annotations

import torch

class Normalizer:
    """
    Min-max normalization of state and action to [-1, 1]; ImageNet normalization of images.

    The same object is used in training and evaluation (guide section 3).
    """

    @classmethod
    def from_stats(cls, stats: dict) -> "Normalizer":
        """
        Build from LeRobot-style stats.

        Args:
            stats: stats['observation.state']['min'|'max'] and stats['action']['min'|'max'],
                array-like of shape (2,) (numpy float64 in dataset.meta.stats and in the
                feature cache). Must also accept the dict returned by state_dict().
        Returns:
            The Normalizer.
        """
        raise NotImplementedError("TODO(core): see guide section 3")

    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize pusher states. Works on any device.

        Args:
            x: (..., 2) raw state.
        Returns:
            (..., 2) in [-1, 1], on the input's device.
        """
        raise NotImplementedError("TODO(core): see guide section 3")

    def normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        """
        Normalize actions. Works on any device.

        Args:
            a: (..., 2) raw action.
        Returns:
            (..., 2) in [-1, 1], on the input's device.
        """
        raise NotImplementedError("TODO(core): see guide section 3")

    def unnormalize_action(self, a_norm: torch.Tensor) -> torch.Tensor:
        """
        Inverse of normalize_action.

        Args:
            a_norm: (..., 2) normalized action.
        Returns:
            (..., 2) raw action (env units), on the input's device.
        """
        raise NotImplementedError("TODO(core): see guide section 3")

    def normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        ImageNet-normalize images.

        Args:
            img: (..., 3, H, W) float in [0, 1].
        Returns:
            ImageNet-normalized images, same shape and device.
        """
        raise NotImplementedError("TODO(core): see guide section 3")

    def state_dict(self) -> dict:
        """
        Stats needed to rebuild this object (used by save_checkpoint).

        Contract: Normalizer.from_stats(n.state_dict()) behaves exactly like n.

        Returns:
            Dict of stats.
        """
        raise NotImplementedError("TODO(core): see guide section 3")
