"""Train one model from a JSON config.

Example:
    python scripts/train.py --config configs/simple_cnn.json --data-root Data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scene_clf.config import Config
from scene_clf.data import make_train_val_loaders
from scene_clf.data.dataset import read_labels_csv
from scene_clf.models import build_model
from scene_clf.seed import set_seed
from scene_clf.training import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a JSON config")
    parser.add_argument("--data-root", help="override config.data.root")
    parser.add_argument("--out-dir", help="override config.train.out_dir")
    parser.add_argument("--epochs", type=int, help="override config.train.epochs")
    parser.add_argument('--resume', action='store_true', help='resume from out_dir/state.pt when present')
    args = parser.parse_args()

    cfg = Config.from_json(args.config)
    if args.data_root:
        cfg.data.root = args.data_root
    if args.out_dir:
        cfg.train.out_dir = args.out_dir
    if args.epochs:
        cfg.train.epochs = args.epochs

    set_seed(cfg.train.seed)
    Path(cfg.train.out_dir).mkdir(parents=True, exist_ok=True)
    cfg.to_json(Path(cfg.train.out_dir) / "config.json")

    train_loader, val_loader = make_train_val_loaders(cfg.data)
    model = build_model(cfg.model)
    parameter_count = model.num_parameters() if hasattr(model, 'num_parameters') else sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'model={cfg.model.name} params={parameter_count:,}')

    train_labels = [label for _, label in read_labels_csv(Path(cfg.data.root) / cfg.data.labels_csv)]
    label_to_idx = train_loader.dataset.class_to_idx
    train_label_idx = [label_to_idx[label] for label in train_labels]

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        cfg.train,
        train_labels=train_label_idx,
        num_classes=cfg.model.num_classes,
    )
    state_path = Path(cfg.train.out_dir) / 'state.pt'
    if args.resume and state_path.exists():
        trainer.load_state()
        print(f'Resuming from epoch {trainer._start_epoch}', flush=True)
    trainer.fit()
    report = trainer.evaluate()
    print(f"best val acc: {report['val_acc']:.4f}")
    print("per-class acc:", {k: round(v, 4) for k, v in report["per_class_acc"].items()})


if __name__ == "__main__":
    main()

