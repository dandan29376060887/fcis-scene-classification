import numpy as np

from scene_clf.training.metrics import accuracy, confusion_matrix, per_class_accuracy


def test_accuracy_perfect_and_zero():
    assert accuracy([0, 1, 2], [0, 1, 2]) == 1.0
    assert accuracy([0, 1, 2], [1, 2, 0]) == 0.0


def test_confusion_and_per_class():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    matrix = confusion_matrix(y_true, y_pred, 3)
    assert matrix.shape == (3, 3)
    assert matrix[0][0] == 1 and matrix[1][1] == 2 and matrix[2][2] == 1
    per = per_class_accuracy(y_true, y_pred, 3)
    assert per[1] == 1.0 and per[0] == 0.5
    assert np.isclose(accuracy(y_true, y_pred), 4 / 6)
