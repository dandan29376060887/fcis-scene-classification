"""SimpleCNN: a plain conv-BN-ReLU-pool stack.

The most direct from-scratch baseline: four convolutional blocks with
progressively more channels, global average pooling, dropout and a
linear head.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import ImageClassifier


class SimpleCNN(ImageClassifier):
    def __init__(self, num_classes: int, base_channels: int = 32, dropout: float = 0.3) -> None:
        super().__init__(num_classes)
        channels = [base_channels * m for m in (1, 2, 4, 8)]  # 32, 64, 128, 256

        blocks = []
        in_ch = 3
        for out_ch in channels:
            blocks += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
