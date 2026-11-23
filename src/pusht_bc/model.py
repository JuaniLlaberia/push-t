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
        """
        Store the encoder and build the MLP head (feat_dim + state_dim) -> hidden -> hidden -> out_dim.

        Args:
            encoder: maps (B, 3, H, W) normalized images to (B, feat_dim), or None for cached features.
            feat_dim: encoder output size.
            state_dim: pusher state size.
            hidden_dim: width of the two hidden layers.
            action_dim: size of one action.
            chunk_size: None for one action per call, K for K actions per call (out_dim = K * action_dim).
        """
        super().__init__()

        self.encoder = encoder
        self.out_dim = action_dim if chunk_size is None else chunk_size * action_dim
        self.head = nn.Sequential(
            nn.Linear(in_features=feat_dim + state_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=self.out_dim)
        )
        self.chunk_size = chunk_size
        self.action_dim = action_dim

    def forward(self, obs_feat_or_image: torch.Tensor, state_norm: torch.Tensor) -> torch.Tensor:
        """
        Predict normalized action(s).

        Args:
            obs_feat_or_image: (B, 512) features if self.encoder is None, else (B, 3, H, W) normalized images.
            state_norm: (B, 2) normalized pusher state.
        Returns:
            (B, 2), or (B, K, 2) when chunk_size=K, in normalized action units.
        """
        if self.encoder is not None:
            x = self.encoder(obs_feat_or_image)
        else:
            x = obs_feat_or_image

        x = torch.cat([x, state_norm], dim=-1)
        x = self.head(x)

        if self.chunk_size is None:
            return x
        else:
            return x.reshape(x.shape[0], self.chunk_size, self.action_dim)

