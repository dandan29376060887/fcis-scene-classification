"""SECNN: conv blocks with squeeze-and-excitation channel attention.

Each block recalibrates its channels with a learned gate: global average
pool -> small dense bottleneck -> sigmoid scaling. Built from explicit
layers like the other architectures.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import ImageClassifier


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.gate(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class SEBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            SqueezeExcite(out_ch),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class SECNN(ImageClassifier):
    def __init__(self, num_classes: int, base_channels: int = 32, dropout: float = 0.3) -> None:
        super().__init__(num_classes)
        widths = [base_channels * m for m in (1, 2, 4, 8)]
        blocks = []
        in_ch = 3
        for out_ch in widths:
            blocks.append(SEBlock(in_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(widths[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
