"""Test-set inference -> submission CSV.

Writes exactly the format the competition scores: a header
``image_id,label`` and one row per id in test.csv order, labels spelled
exactly as the class table. Optional test-time augmentation (horizontal
flip averaging) can be switched on; it uses no extra data, so it stays
within the rules.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Sequence

import torch
from torch.utils.data import DataLoader

from ..constants import IDX_TO_CLASS
from ..seed import resolve_device


@torch.no_grad()
def predict_labels(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta_flip: bool = False,
    tta_8way: bool = False,
) -> List[str]:
    """Predict one class name per loader item, in loader order."""
    model.eval()
    predictions: List[str] = []
    for images, _ in loader:
        images = images.to(device)
        if tta_8way:
            # 8 dihedral orientations (4 rotations * 2 flips)
            preds = torch.zeros(images.shape[0], 11, device=device)
            for k in range(4):
                rotated = torch.rot90(images, k, [2, 3])
                preds += model(rotated).softmax(dim=1)
                preds += model(torch.flip(rotated, dims=[3])).softmax(dim=1)
            logits = preds
        elif tta_flip:
            logits = model(images) + model(torch.flip(images, dims=[3]))
        else:
            logits = model(images)
        predictions.extend(IDX_TO_CLASS[i] for i in logits.argmax(dim=1).tolist())
    return predictions


def write_submission(path: str | Path, image_ids: Sequence[str], labels: Sequence[str]) -> None:
    if len(image_ids) != len(labels):
        raise ValueError("one label required per image_id")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "label"])
        writer.writerows(zip(image_ids, labels))


def predict_to_csv(
    model: torch.nn.Module,
    loader: DataLoader,
    image_ids: Sequence[str],
    out_path: str | Path,
    device: str = "auto",
    tta_flip: bool = False,
) -> None:
    resolved = resolve_device(device)
    labels = predict_labels(model.to(resolved), loader, resolved, tta_flip=tta_flip)
    write_submission(out_path, image_ids, labels)
