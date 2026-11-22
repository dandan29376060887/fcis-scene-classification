"""Datasets for the competition images.

The labeled training set is described by ``labels.csv`` (image_id,label)
plus a directory of ``<image_id>.jpg`` files. The unlabeled test set is
described by ``test.csv`` (image_id) the same way.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from ..constants import CLASS_TO_IDX


def read_labels_csv(path: str | Path) -> List[Tuple[str, str]]:
    """Read labels.csv -> [(image_id, label), ...] preserving file order."""
    with open(path, newline="") as f:
        return [(row["image_id"], row["label"]) for row in csv.DictReader(f)]


def read_id_csv(path: str | Path) -> List[str]:
    """Read test.csv -> [image_id, ...] preserving file order."""
    with open(path, newline="") as f:
        return [row["image_id"] for row in csv.DictReader(f)]


class SceneDataset(Dataset):
    """Labeled images.

    ``samples`` is a list of (image_id, label) pairs; keeping it outside
    the dataset lets the caller control splitting while the dataset only
    cares about loading and transforming.
    """

    def __init__(
        self,
        image_dir: str | Path,
        samples: List[Tuple[str, str]],
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.samples = samples
        self.transform = transform
        self.class_to_idx = class_to_idx or CLASS_TO_IDX
        unknown = {label for _, label in samples} - set(self.class_to_idx)
        if unknown:
            raise ValueError(f"Unknown labels in samples: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_id, label = self.samples[index]
        image = Image.open(self.image_dir / f"{image_id}.jpg").convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_idx[label]


class TestDataset(Dataset):
    """Unlabeled test images; yields (image, image_id) so predictions can
    be joined back onto the submission file."""

    def __init__(
        self,
        image_dir: str | Path,
        image_ids: List[str],
        transform: Optional[Callable] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.image_ids = image_ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        image_id = self.image_ids[index]
        image = Image.open(self.image_dir / f"{image_id}.jpg").convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_id
