from __future__ import annotations

import torch

def bc_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, kind: str = "mse") -> torch.Tensor:
    """
    Behavior-cloning loss, averaged over real (unmasked) elements only.

    Args:
        pred: (B, 2) or (B, K, 2).
        target: same shape as pred.
        mask: (B,) or (B, K) bool, True = real target; None = all real.
        kind: "mse" or "l1".
    Returns:
        Scalar tensor; 0 if there are no real elements.
    """
    if kind == "mse":
        err = (target - pred) ** 2
    elif kind == "l1":
        err = torch.abs(target - pred)
    else:
        raise ValueError("Loss 'kind' must be either 'mse' or 'l1'")

    if mask is not None:
        err = err[mask]

    if err.numel() == 0:
        return err.sum()

    return err.mean()