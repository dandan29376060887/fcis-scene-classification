import csv
import json
from pathlib import Path

import torch

from scene_clf.data import make_test_loader, make_train_val_loaders
from scene_clf.data.dataset import read_id_csv
from scene_clf.inference import predict_to_csv
from scene_clf.models import build_model
from scene_clf.training import Trainer


def test_trainer_smoke_run_and_checkpoints(small_cfg):
    train_loader, val_loader = make_train_val_loaders(small_cfg.data)
    model = build_model(small_cfg.model)
    trainer = Trainer(model, train_loader, val_loader, small_cfg.train, num_classes=4)
    history = trainer.fit()
    assert len(history) >= 1
    out = Path(small_cfg.train.out_dir)
    assert (out / "best.pt").exists()
    assert (out / "last.pt").exists()
    saved_history = json.loads((out / "history.json").read_text())
    assert saved_history[0]["epoch"] == 1
    report = trainer.evaluate()
    assert 0.0 <= report["val_acc"] <= 1.0
    assert len(report["per_class_acc"]) == 4


def test_predict_writes_valid_submission(small_cfg, mini_dataset):
    train_loader, val_loader = make_train_val_loaders(small_cfg.data)
    model = build_model(small_cfg.model)
    trainer = Trainer(model, train_loader, val_loader, small_cfg.train, num_classes=4)
    trainer.fit()

    test_loader = make_test_loader(small_cfg.data)
    image_ids = read_id_csv(mini_dataset / "test.csv")
    out_csv = mini_dataset.parent / "submission.csv"
    predict_to_csv(model, test_loader, image_ids, out_csv, device="cpu")

    with open(out_csv, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["image_id", "label"]
    assert len(rows) == len(image_ids) + 1
    assert {r[0] for r in rows[1:]} == set(image_ids)  # every id exactly once


def test_predict_tta_path_runs(small_cfg, mini_dataset):
    model = build_model(small_cfg.model)
    test_loader = make_test_loader(small_cfg.data)
    image_ids = read_id_csv(mini_dataset / "test.csv")
    out_csv = mini_dataset.parent / "submission_tta.csv"
    predict_to_csv(model, test_loader, image_ids, out_csv, device="cpu", tta_flip=True)
    assert out_csv.exists()
