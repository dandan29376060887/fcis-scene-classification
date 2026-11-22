"""Produce a competition submission CSV from a trained checkpoint.

Example:
    python scripts/predict.py --config configs/simple_cnn.json \
        --checkpoint runs/run/best.pt --data-root Data --out submission.csv --tta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from scene_clf.config import Config
from scene_clf.data import make_test_loader
from scene_clf.data.dataset import read_id_csv
from scene_clf.inference import predict_to_csv
from scene_clf.models import build_model
from scene_clf.seed import resolve_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON config used for training")
    parser.add_argument("--checkpoint", required=True, help="path to best.pt")
    parser.add_argument("--data-root", help="override config.data.root")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--tta", action="store_true", help="average with horizontal-flip prediction")
    args = parser.parse_args()

    cfg = Config.from_json(args.config)
    if args.data_root:
        cfg.data.root = args.data_root
    set_seed(cfg.train.seed)

    model = build_model(cfg.model)
    device = resolve_device(cfg.train.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state"])

    test_loader = make_test_loader(cfg.data)
    image_ids = read_id_csv(Path(cfg.data.root) / cfg.data.test_csv)
    predict_to_csv(model, test_loader, image_ids, args.out, device=str(device), tta_flip=args.tta)
    print(f"wrote {len(image_ids)} predictions to {args.out}")


if __name__ == "__main__":
    main()
