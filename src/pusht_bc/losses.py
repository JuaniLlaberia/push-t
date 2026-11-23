from __future__ import annotations

import torch

def bc_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, kind: str = "mse") -> torch.Tensor:
    """
    Behavior-cloning loss: mean over real elements only (guide section 5).

    Args:
        pred: (B, 2) or (B, K, 2).
        target: same shape as pred.
        mask: (B,) or (B, K) bool, True = real target; None = all real.
        kind: "mse" or "l1".
    Returns:
        Scalar tensor; 0 if there are no real elements.
    """
    raise NotImplementedError("TODO(core): see guide section 5")
