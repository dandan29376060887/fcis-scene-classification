"""Training loop, kept free of data/model construction concerns.

The Trainer receives a model, two loaders and an optimiser setup, so any
model from the registry and any loader can be plugged in. It owns:
epochs, optimisation, validation, checkpointing (best + last), early
stopping and a JSON history. Mixed precision is used on CUDA only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from ..config import TrainConfig
from ..seed import resolve_device
from .metrics import accuracy, per_class_accuracy


def build_optimizer(cfg: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    raise KeyError(f"Unknown optimizer {cfg.optimizer!r}")


def build_scheduler(cfg: TrainConfig, optimizer: torch.optim.Optimizer):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)
    if cfg.scheduler == "none":
        return None
    raise KeyError(f"Unknown scheduler {cfg.scheduler!r}")


def class_weights_from_labels(labels: List[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights so rare classes count as much as common ones."""
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        cfg: TrainConfig,
        train_labels: Optional[List[int]] = None,
        num_classes: int = 11,
    ) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes

        if cfg.class_weighted:
            if train_labels is None:
                raise ValueError("class_weighted=True requires train_labels")
            weight = class_weights_from_labels(train_labels, num_classes).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = build_optimizer(cfg, model)
        self.scheduler = build_scheduler(cfg, self.optimizer)
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.history: List[Dict] = []
        self.best_val_acc = -1.0
        self._epochs_without_improvement = 0
        self._start_epoch = 1
        self._last_epoch = 0

    # ------------------------------------------------------------------
    def _run_epoch(self, train: bool) -> Tuple[float, float]:
        model = self.model
        model.train() if train else model.eval()
        loader = self.train_loader if train else self.val_loader
        total_loss = 0.0
        y_true: List[int] = []
        y_pred: List[int] = []
        use_amp = train and self.device.type == "cuda"

        with torch.set_grad_enabled(train):
            for images, targets in loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, enabled=use_amp):
                    logits = model(images)
                    loss = self.criterion(logits, targets)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()
                total_loss += loss.item() * len(targets)
                y_true.extend(targets.tolist())
                y_pred.extend(logits.argmax(dim=1).tolist())
        return total_loss / len(y_true), accuracy(y_true, y_pred)

    # ------------------------------------------------------------------
    def _save_checkpoint(self, name: str) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "model_num_classes": self.num_classes,
                "val_acc": self.best_val_acc,
            },
            self.out_dir / name,
        )

    def fit(self, max_seconds: float | None = None) -> List[Dict]:
        started_fit = time.time()
        for epoch in range(self._start_epoch, self.cfg.epochs + 1):
            started = time.time()
            gen = getattr(self.train_loader, 'generator', None)
            if gen is not None:
                gen.manual_seed(self.cfg.seed + epoch)
            train_loss, train_acc = self._run_epoch(train=True)
            val_loss, val_acc = self._run_epoch(train=False)

            if self.scheduler is not None:
                if self.cfg.scheduler == "plateau":
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "train_acc": round(train_acc, 5),
                "val_loss": round(val_loss, 5),
                "val_acc": round(val_acc, 5),
                "lr": self.optimizer.param_groups[0]["lr"],
                "seconds": round(time.time() - started, 1),
            }
            self._last_epoch = epoch
            self.history.append(record)
            print(
                f"epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.4f} "
                f"| val loss {val_loss:.4f} acc {val_acc:.4f} | {record['seconds']}s",
                flush=True,
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._epochs_without_improvement = 0
                self._save_checkpoint("best.pt")
            else:
                self._epochs_without_improvement += 1
            self._save_checkpoint("last.pt")
            (self.out_dir / "history.json").write_text(json.dumps(self.history, indent=2))

            self._save_state()
            if self._epochs_without_improvement >= self.cfg.early_stopping_patience:
                print(f"early stopping after epoch {epoch}", flush=True)
                break
            if max_seconds is not None and time.time() - started_fit > max_seconds:
                print(f"stopping chunk after epoch {epoch} (time budget)", flush=True)
                break
        return self.history

    # ------------------------------------------------------------------
    def _save_state(self) -> None:
        import numpy as _np, random as _random
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
                "epoch": self._last_epoch,
                "history": self.history,
                "best_val_acc": self.best_val_acc,
                "epochs_without_improvement": self._epochs_without_improvement,
                "torch_rng": torch.get_rng_state(),
                "np_rng": _np.random.get_state(),
                "py_rng": _random.getstate(),
            },
            self.out_dir / "state.pt",
        )

    def load_state(self) -> None:
        import numpy as _np, random as _random
        state = torch.load(self.out_dir / "state.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        if self.scheduler and state["scheduler_state"]:
            self.scheduler.load_state_dict(state["scheduler_state"])
        self._start_epoch = state["epoch"] + 1
        self.history = state["history"]
        self.best_val_acc = state["best_val_acc"]
        self._epochs_without_improvement = state["epochs_without_improvement"]
        torch.set_rng_state(state["torch_rng"])
        _np.random.set_state(state["np_rng"])
        _random.setstate(state["py_rng"])

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Validation metrics of the *best* checkpoint (reloaded)."""
        best = self.out_dir / "best.pt"
        if best.exists():
            state = torch.load(best, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state["model_state"])
        self.model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []
        for images, targets in self.val_loader:
            logits = self.model(images.to(self.device))
            y_true.extend(targets.tolist())
            y_pred.extend(logits.argmax(dim=1).tolist())
        return {
            "val_acc": accuracy(y_true, y_pred),
            "per_class_acc": per_class_accuracy(y_true, y_pred, self.num_classes),
        }
