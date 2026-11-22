"""Augmentation / preprocessing pipelines.

Training augmentation targets the variation named in the competition
brief (lighting, season, colour, viewpoint) without inventing scenes the
data never shows: geometric flips/rotations and moderate crops and
colour jitter.
"""

from __future__ import annotations

from torchvision import transforms

from ..constants import IMAGENET_MEAN, IMAGENET_STD


def build_train_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
