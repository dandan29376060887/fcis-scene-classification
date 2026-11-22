from scene_clf.data import (
    build_eval_transforms,
    build_train_transforms,
    make_test_loader,
    make_train_val_loaders,
    stratified_split,
)
from scene_clf.data.dataset import read_labels_csv


def test_stratified_split_is_stratified_and_disjoint():
    labels = ["a"] * 10 + ["b"] * 6 + ["c"] * 4
    train_idx, val_idx = stratified_split(labels, 0.25, seed=7)
    assert set(train_idx).isdisjoint(val_idx)
    assert len(train_idx) + len(val_idx) == len(labels)
    val_labels = [labels[i] for i in val_idx]
    assert set(val_labels) == {"a", "b", "c"}  # every class represented


def test_stratified_split_rejects_bad_fraction():
    import pytest

    with pytest.raises(ValueError):
        stratified_split(["a", "b"], 1.0, seed=0)


def test_transform_shapes():
    import torch
    from PIL import Image

    img = Image.new("RGB", (48, 64), color=(10, 20, 30))
    train_tf = build_train_transforms(32)
    eval_tf = build_eval_transforms(32)
    assert train_tf(img).shape == (3, 32, 32)
    assert eval_tf(img).shape == (3, 32, 32)
    assert isinstance(train_tf(img), torch.Tensor)


def test_loaders_cover_train_and_val(mini_dataset, small_cfg):
    train_loader, val_loader = make_train_val_loaders(small_cfg.data)
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    assert n_train + n_val == 24
    assert n_val >= 4  # 4 classes x at least 1
    batch_images, batch_labels = next(iter(train_loader))
    assert batch_images.shape[1:] == (3, 32, 32)


def test_test_loader_keeps_test_csv_order(mini_dataset, small_cfg):
    loader = make_test_loader(small_cfg.data)
    _, ids = next(iter(loader))
    assert list(ids) == [f"test{i:04d}" for i in range(6)][: len(ids)]
