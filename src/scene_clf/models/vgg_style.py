"""VGGStyleCNN: two convs per stage before pooling, VGG-flavoured.

Deeper per-stage feature extraction than SimpleCNN; every layer is still
defined here explicitly.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import ImageClassifier


def _stage(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class VGGStyleCNN(ImageClassifier):
    def __init__(self, num_classes: int, base_channels: int = 32, dropout: float = 0.3) -> None:
        super().__init__(num_classes)
        widths = [base_channels * m for m in (1, 2, 4)]  # 32, 64, 128
        stages = []
        in_ch = 3
        for out_ch in widths:
            stages.append(_stage(in_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(widths[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
