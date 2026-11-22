"""ResidualCNN: residual blocks with skip connections, defined from scratch.

Deeper networks train better when blocks can learn a residual mapping
F(x) added to a (possibly projected) identity of x. This is the same
idea as a ResNet, but the block is written out here layer by layer as
the rules require.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import ImageClassifier


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection around them."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResidualCNN(ImageClassifier):
    def __init__(self, num_classes: int, base_channels: int = 32, dropout: float = 0.3) -> None:
        super().__init__(num_classes)
        widths = [base_channels * m for m in (1, 2, 4, 8)]  # 32, 64, 128, 256
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        stages = []
        in_ch = widths[0]
        for i, out_ch in enumerate(widths):
            stride = 1 if i == 0 else 2
            stages.append(ResidualBlock(in_ch, out_ch, stride=stride))
            stages.append(ResidualBlock(out_ch, out_ch, stride=1))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(widths[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(x)))
