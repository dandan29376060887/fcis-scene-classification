"""Model registry.

``build_model`` is the single entry point the trainer uses, so swapping
architectures is a config change, not a code change. The from-scratch
architectures satisfy the competition requirement of at least four
models with self-defined layers, and ``vit_scratch`` covers the
amendment that one of the four must be a transformer (self-attention
implemented from primitive Linear layers, no ready-made model classes).
``resnet18_transfer`` is a legal bonus (pretrained weights are
explicitly allowed) but does NOT count toward the four.
"""

from __future__ import annotations

from typing import Callable, Dict

from torch import nn

from ..config import ModelConfig
from .base import ImageClassifier
from .residual_cnn import ResidualCNN
from .se_cnn import SECNN
from .simple_cnn import SimpleCNN
from .vgg_style import VGGStyleCNN
from .vit_scratch import ScratchViT


def _build_scratch(cls):
    def builder(cfg: ModelConfig) -> ImageClassifier:
        return cls(
            num_classes=cfg.num_classes,
            base_channels=cfg.base_channels,
            dropout=cfg.dropout,
        )

    return builder


def _build_resnet18_transfer(cfg: ModelConfig) -> ImageClassifier:
    """torchvision ResNet-18 with a replaced head. Transfer learning is
    allowed by the rules; it does not count as a from-scratch model."""
    from torchvision import models

    weights = models.ResNet18_Weights.DEFAULT if cfg.pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, cfg.num_classes)
    return net  # type: ignore[return-value]


def _build_vit_scratch(cfg: ModelConfig) -> ImageClassifier:
    extra = dict(cfg.extra)
    return ScratchViT(
        num_classes=cfg.num_classes,
        base_channels=cfg.base_channels,
        dropout=cfg.dropout,
        patch_size=extra.get("patch_size", 16),
        depth=extra.get("depth", 4),
        heads=extra.get("heads", 4),
        mlp_ratio=extra.get("mlp_ratio", 4.0),
        image_size=extra.get("image_size", 224),
    )


MODEL_REGISTRY: Dict[str, Callable[[ModelConfig], ImageClassifier]] = {
    "simple_cnn": _build_scratch(SimpleCNN),
    "vgg_style": _build_scratch(VGGStyleCNN),
    "residual_cnn": _build_scratch(ResidualCNN),
    "se_cnn": _build_scratch(SECNN),
    "resnet18_transfer": _build_resnet18_transfer,
    "vit_scratch": _build_vit_scratch,
}

SCRATCH_MODELS = ["simple_cnn", "vgg_style", "residual_cnn", "se_cnn", "vit_scratch"]


def build_model(cfg: ModelConfig) -> ImageClassifier:
    if cfg.name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {cfg.name!r}. Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[cfg.name](cfg)

