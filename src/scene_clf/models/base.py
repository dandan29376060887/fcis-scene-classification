"""Common base class for the from-scratch classifiers."""

from __future__ import annotations

import torch
from torch import nn


class ImageClassifier(nn.Module):
    """Maps an image batch (B, 3, H, W) to class logits (B, num_classes).

    All competition models are built from explicitly defined layers
    (convolutions, pooling, normalisation, dense) rather than ready-made
    architecture classes, as the rules require for at least four models.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)
