"""Synthetic mini-dataset: deterministic coloured-noise JPEGs written to
a tmp dir in the real competition layout (data/, test/, labels.csv,
test.csv). Tests never touch the real competition data.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scene_clf.constants import CLASSES  # noqa: E402


def _write_image(path: Path, seed: int, size: int = 32) -> None:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(3,), dtype=np.uint8)
    noise = rng.integers(0, 32, size=(size, size, 3), dtype=np.uint8)
    arr = base.reshape(1, 1, 3) + noise
    Image.fromarray(arr.astype(np.uint8), "RGB").save(path, "JPEG")


@pytest.fixture()
def mini_dataset(tmp_path: Path) -> Path:
    """A 4-class, 24-image training set and a 6-image test set."""
    root = tmp_path / "Data"
    train_dir = root / "data"
    test_dir = root / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    classes = CLASSES[:4]
    train_rows = []
    idx = 0
    for cls in classes:
        for _ in range(6):
            image_id = f"train{idx:04d}"
            _write_image(train_dir / f"{image_id}.jpg", seed=idx)
            train_rows.append((image_id, cls))
            idx += 1

    with open(root / "labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "label"])
        writer.writerows(train_rows)

    test_ids = []
    for i in range(6):
        image_id = f"test{i:04d}"
        _write_image(test_dir / f"{image_id}.jpg", seed=10_000 + i)
        test_ids.append(image_id)
    with open(root / "test.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id"])
        writer.writerows([(i,) for i in test_ids])

    return root


@pytest.fixture()
def small_cfg(mini_dataset: Path):
    from scene_clf.config import Config, DataConfig, ModelConfig, TrainConfig

    return Config(
        data=DataConfig(
            root=str(mini_dataset),
            image_size=32,
            val_fraction=0.25,
            batch_size=8,
            num_workers=0,
            seed=123,
            augment=True,
        ),
        model=ModelConfig(name="simple_cnn", num_classes=4, dropout=0.1, base_channels=8),
        train=TrainConfig(
            epochs=2,
            lr=1e-3,
            scheduler="cosine",
            early_stopping_patience=5,
            device="cpu",
            out_dir=str(mini_dataset.parent / "runs" / "test"),
            seed=123,
        ),
    )
