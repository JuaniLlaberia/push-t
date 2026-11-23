from __future__ import annotations

import torch
from torch import nn

class ResNet18Encoder(nn.Module):
    """
    ResNet-18 trunk with global average pooling: (B, 3, H, W) -> (B, 512).

    Input must already be ImageNet-normalized (see Normalizer.normalize_image).
    When frozen, every parameter has requires_grad=False and the trunk stays in eval mode
    whatever .train() is called on this module or on any parent, so BatchNorm running
    statistics never update; the forward pass then runs without building a graph.
    The trunk is always stored as self.net, so frozen and trainable encoders have the
    same state-dict keys (checkpoints rely on this).

    Args:
        pretrained: load torchvision's ImageNet weights.
        frozen: freeze the encoder as described above.
    """

    out_dim = 512

    def __init__(self, pretrained: bool = True, frozen: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        net.fc = nn.Identity()
        self.net = net
        self.frozen = frozen
        if frozen:
            for p in self.net.parameters():
                p.requires_grad_(False)
            self.net.eval()

    def train(self, mode: bool = True) -> "ResNet18Encoder":
        """
        Set training mode; a frozen trunk stays in eval mode.

        Args:
            mode: training mode for this module.
        Returns:
            self.
        """
        super().train(mode)
        if self.frozen:
            self.net.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images.

        Args:
            x: (B, 3, H, W) ImageNet-normalized images, H, W >= 64.
        Returns:
            (B, 512) features.
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected (B, 3, H, W), got {tuple(x.shape)}")
        if min(x.shape[-2:]) < 64:
            raise ValueError(f"image side must be >= 64, got {tuple(x.shape[-2:])}")
        if self.frozen:
            with torch.no_grad():
                return self.net(x)
        return self.net(x)

def make_resnet18_encoder(pretrained: bool = True, frozen: bool = True) -> nn.Module:
    """
    torchvision ResNet-18 with the classifier removed.

    Args:
        pretrained: load ImageNet weights.
        frozen: freeze parameters and keep BatchNorm in eval mode.
    Returns:
        Module mapping (B, 3, H, W) normalized images to (B, 512).
    """
    return ResNet18Encoder(pretrained=pretrained, frozen=frozen)
