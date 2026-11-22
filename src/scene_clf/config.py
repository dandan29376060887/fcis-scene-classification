"""Configuration objects.

Everything that defines an experiment (data layout, model choice,
training hyper-parameters) lives in these dataclasses. A config can be
serialised to / from JSON so a run is fully reproducible from the file
checked into version control.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class DataConfig:
    """Where the data lives and how it is fed to the model."""

    root: str = "Data"                # dir containing data/, test/, labels.csv, test.csv
    train_dir: str = "data"           # subdir of root with training images
    test_dir: str = "test"            # subdir of root with test images
    labels_csv: str = "labels.csv"
    test_csv: str = "test.csv"
    image_size: int = 224             # model input resolution
    val_fraction: float = 0.15        # stratified hold-out fraction
    batch_size: int = 32
    num_workers: int = 2
    seed: int = 42
    augment: bool = True              # train-time augmentation on/off


@dataclass
class ModelConfig:
    """Which architecture to build and its knobs."""

    name: str = "simple_cnn"
    num_classes: int = 11
    dropout: float = 0.3
    base_channels: int = 32
    pretrained: bool = False          # only used by transfer-learning models
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    """Optimisation settings."""

    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"          # adamw | sgd
    momentum: float = 0.9             # sgd only
    scheduler: str = "cosine"         # cosine | plateau | none
    early_stopping_patience: int = 7
    class_weighted: bool = False      # weight CE by inverse class frequency
    grad_clip: float = 0.0            # 0 disables clipping
    device: str = "auto"              # auto | cpu | cuda
    out_dir: str = "runs/run"
    seed: int = 42


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text())
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
        )
