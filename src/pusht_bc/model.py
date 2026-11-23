from __future__ import annotations

import torch
from torch import nn

class BCPolicy(nn.Module):
    """
    Deterministic BC policy: image features (+ pusher state) -> normalized action(s) (guide sections 4 and 10).

    Attribute convention (used by checkpoint loading):
        self.encoder: nn.Module or None (None = inputs are cached (B, 512) features).
        self.head: nn.Module holding every parameter that is not the encoder's.
        self.chunk_size: int or None, as passed to __init__.
    save_checkpoint stores only the head (and a trained encoder); load_policy_for_eval
    rebuilds BCPolicy(encoder=..., hidden_dim=cfg.hidden_dim, chunk_size=cfg.chunk_size)
    and loads those weights back.

    Args:
        encoder: maps (B, 3, H, W) normalized images to (B, feat_dim), or None.
        feat_dim: encoder output size.
        state_dim: pusher state size.
        hidden_dim: hidden width.
        action_dim: action size.
        chunk_size: None for one action per call, K for K actions per call (section 10).
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        feat_dim: int = 512,
        state_dim: int = 2,
        hidden_dim: int = 256,
        action_dim: int = 2,
        chunk_size: int | None = None,
    ):
        raise NotImplementedError("TODO(core): see guide sections 4 and 10")

    def forward(self, obs_feat_or_image: torch.Tensor, state_norm: torch.Tensor) -> torch.Tensor:
        """
        Predict normalized action(s).

        Args:
            obs_feat_or_image: (B, 512) features if self.encoder is None, else (B, 3, H, W) normalized images.
            state_norm: (B, 2) normalized pusher state.
        Returns:
            (B, 2), or (B, K, 2) when chunk_size=K, in normalized action units.
        """
        raise NotImplementedError("TODO(core): see guide sections 4 and 10")
