"""Pure metric functions. No model or loader dependencies."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """Fraction of correct predictions. This is the competition metric."""
    if len(y_true) == 0:
        raise ValueError("accuracy of an empty set is undefined")
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int) -> np.ndarray:
    """rows = true class, cols = predicted class."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def per_class_accuracy(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[int, float]:
    """Per-class recall; useful to spot classes the model confuses."""
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    result: Dict[int, float] = {}
    for c in range(num_classes):
        total = matrix[c].sum()
        result[c] = float(matrix[c][c] / total) if total else 0.0
    return result
