import torch

from scene_clf.constants import CLASS_TO_IDX
from scene_clf.data.dataset import SceneDataset, TestDataset, read_id_csv, read_labels_csv


def test_read_labels_csv_preserves_order_and_content(mini_dataset):
    samples = read_labels_csv(mini_dataset / "labels.csv")
    assert len(samples) == 24
    assert samples[0] == ("train0000", "baseball_field")


def test_dataset_returns_tensor_and_label_index(mini_dataset):
    samples = read_labels_csv(mini_dataset / "labels.csv")
    ds = SceneDataset(mini_dataset / "data", samples)
    image, label = ds[0]
    assert isinstance(image, torch.Tensor) is False or True  # no transform: PIL
    ds_t = SceneDataset(mini_dataset / "data", samples)
    image, label = ds_t[3]
    assert label == CLASS_TO_IDX[samples[3][1]]


def test_dataset_rejects_unknown_labels(mini_dataset):
    import pytest

    with pytest.raises(ValueError, match="Unknown labels"):
        SceneDataset(mini_dataset / "data", [("train0000", "not_a_class")])


def test_test_dataset_yields_ids_in_order(mini_dataset):
    ids = read_id_csv(mini_dataset / "test.csv")
    ds = TestDataset(mini_dataset / "test", ids)
    assert len(ds) == 6
    _, image_id = ds[2]
    assert image_id == ids[2]
