"""Splitting and DataLoader construction."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import DataConfig
from .dataset import SceneDataset, TestDataset, read_id_csv, read_labels_csv
from .transforms import build_eval_transforms, build_train_transforms


def stratified_split(
    labels: Sequence[str], val_fraction: float, seed: int
) -> Tuple[List[int], List[int]]:
    """Class-stratified train/val index split.

    Every class contributes ``val_fraction`` of its samples to the
    validation set (rounded), so rare and common classes keep the same
    proportions in both splits.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for cls in sorted(set(labels)):
        cls_idx = np.flatnonzero(labels_arr == cls)
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_fraction)))
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def make_train_val_loaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader]:
    """Build the train and validation loaders from a DataConfig."""
    root = Path(cfg.root)
    samples = read_labels_csv(root / cfg.labels_csv)
    labels = [label for _, label in samples]
    train_idx, val_idx = stratified_split(labels, cfg.val_fraction, cfg.seed)

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    image_dir = root / cfg.train_dir
    train_tf = (
        build_train_transforms(cfg.image_size)
        if cfg.augment
        else build_eval_transforms(cfg.image_size)
    )
    train_ds = SceneDataset(image_dir, train_samples, transform=train_tf)
    val_ds = SceneDataset(image_dir, val_samples, transform=build_eval_transforms(cfg.image_size))

    generator = np.random.Generator(np.random.PCG64(cfg.seed))
    torch_seed = int(generator.integers(0, 2**31 - 1))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(torch_seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def make_test_loader(cfg: DataConfig) -> DataLoader:
    """Build the test loader in test.csv order."""
    root = Path(cfg.root)
    image_ids = read_id_csv(root / cfg.test_csv)
    test_ds = TestDataset(
        root / cfg.test_dir, image_ids, transform=build_eval_transforms(cfg.image_size)
    )
    return DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
