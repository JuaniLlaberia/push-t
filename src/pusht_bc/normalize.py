from __future__ import annotations

import torch

class Normalizer:
    """
    Min-max normalization of state and action to [-1, 1]; ImageNet normalization of images.

    The same stats are used in training and evaluation (they are saved in each checkpoint).
    """
    def __init__(self,
                 state_min,
                 state_max,
                 action_min,
                 action_max):
        """
        Store the min-max stats as float32 tensors and set the ImageNet constants.

        Args:
            state_min: (2,) array-like, per-dimension minimum of observation.state.
            state_max: (2,) array-like, per-dimension maximum of observation.state.
            action_min: (2,) array-like, per-dimension minimum of action.
            action_max: (2,) array-like, per-dimension maximum of action.
        """
        self.state_min = torch.as_tensor(state_min, dtype=torch.float32)
        self.state_max = torch.as_tensor(state_max, dtype=torch.float32)
        self.action_min = torch.as_tensor(action_min, dtype=torch.float32)
        self.action_max = torch.as_tensor(action_max, dtype=torch.float32)
        self.imgnet_mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.imgnet_std = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

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
        return cls(stats["observation.state"]["min"],
                   stats["observation.state"]["max"],
                   stats["action"]["min"],
                   stats["action"]["max"])

    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize pusher states. Works on any device.

        Args:
            x: (..., 2) raw state.
        Returns:
            (..., 2) in [-1, 1], on the input's device.
        """
        EPSILON = 1e-8
        x_min = self.state_min.to(x)
        x_max = self.state_max.to(x)

        x_norm = 2 * ((x - x_min) / (x_max - x_min + EPSILON)) - 1

        return x_norm

    def normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        """
        Normalize actions. Works on any device.

        Args:
            a: (..., 2) raw action.
        Returns:
            (..., 2) in [-1, 1], on the input's device.
        """
        EPSILON = 1e-8
        a_min = self.action_min.to(a)
        a_max = self.action_max.to(a)

        a_norm = 2 * ((a - a_min) / (a_max - a_min + EPSILON)) - 1

        return a_norm

    def unnormalize_action(self, a_norm: torch.Tensor) -> torch.Tensor:
        """
        Inverse of normalize_action.

        Args:
            a_norm: (..., 2) normalized action.
        Returns:
            (..., 2) raw action (env units), on the input's device.
        """
        EPSILON = 1e-8
        a_min = self.action_min.to(a_norm)
        a_max = self.action_max.to(a_norm)

        a = ((a_norm + 1) / 2) * (a_max - a_min + EPSILON) + a_min

        return a

    def normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        ImageNet-normalize images.

        Args:
            img: (..., 3, H, W) float in [0, 1].
        Returns:
            ImageNet-normalized images, same shape and device.
        """
        return (img -  self.imgnet_mean.to(img)) / self.imgnet_std.to(img)

    def state_dict(self) -> dict:
        """
        Stats needed to rebuild this object (used by save_checkpoint).

        Contract: Normalizer.from_stats(n.state_dict()) behaves exactly like n.

        Returns:
            Dict of stats.
        """
        return {
            "observation.state": {"min": self.state_min.clone(), "max": self.state_max.clone()},
            "action": {"min": self.action_min.clone(), "max": self.action_max.clone()}
        }
