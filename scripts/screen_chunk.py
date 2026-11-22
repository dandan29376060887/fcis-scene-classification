"""Run the 4-model screening comparison in resumable time-budgeted chunks.

Usage:
    python scripts/screen_chunk.py --data-root Data --workdir screening \
        --epochs 6 --image-size 64 --budget-seconds 95

Each invocation trains every unfinished model until the time budget is
hit, checkpointing full state (model/optimizer/scheduler/RNG/history)
after every epoch, so the next invocation resumes exactly. Deterministic:
the same split (seed) and per-epoch shuffle seed are used on every run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scene_clf.config import Config, DataConfig, ModelConfig, TrainConfig
from scene_clf.data import make_train_val_loaders
from scene_clf.models import SCRATCH_MODELS, build_model
from scene_clf.seed import set_seed
from scene_clf.training import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--models", nargs="*", default=SCRATCH_MODELS)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--budget-seconds", type=float, default=95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    results_path = workdir / "screening_results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    deadline = time.time() + args.budget_seconds

    for name in args.models:
        if name in results and results[name].get("done"):
            continue
        cfg = Config(
            data=DataConfig(
                root=args.data_root, image_size=args.image_size,
                val_fraction=0.15, batch_size=args.batch_size,
                num_workers=0, seed=args.seed, augment=True,
            ),
            model=ModelConfig(name=name, num_classes=11, dropout=0.3, base_channels=32),
            train=TrainConfig(
                epochs=args.epochs, lr=1e-3, weight_decay=1e-4,
                optimizer="adamw", scheduler="cosine",
                early_stopping_patience=3, grad_clip=1.0,
                device="cpu", out_dir=str(workdir / name), seed=args.seed,
            ),
        )
        train_loader, val_loader = make_train_val_loaders(cfg.data)
        model = build_model(cfg.model)
        trainer = Trainer(model, train_loader, val_loader, cfg.train, num_classes=11)
        state_file = Path(cfg.train.out_dir) / "state.pt"
        if state_file.exists():
            trainer.load_state()
            print(f"resuming {name} from epoch {trainer._start_epoch}", flush=True)
        else:
            set_seed(cfg.train.seed)
            Path(cfg.train.out_dir).mkdir(parents=True, exist_ok=True)
            cfg.to_json(Path(cfg.train.out_dir) / "config.json")
            trainer._save_state()

        remaining = deadline - time.time()
        if remaining <= 5:
            print("time budget exhausted, stopping", flush=True)
            break
        trainer.fit(max_seconds=max(remaining - 5, 10))

        if trainer._last_epoch >= cfg.train.epochs or trainer._epochs_without_improvement >= cfg.train.early_stopping_patience:
            report = trainer.evaluate()
            results[name] = {
                "done": True,
                "best_val_acc": report["val_acc"],
                "per_class_acc": {str(k): round(v, 4) for k, v in report["per_class_acc"].items()},
                "epochs_run": trainer._last_epoch,
            }
            results_path.write_text(json.dumps(results, indent=2))
            print(f"DONE {name}: best val acc {report['val_acc']}", flush=True)
        else:
            print(f"chunk ended mid-model {name} at epoch {trainer._last_epoch}", flush=True)
            break

    results_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
