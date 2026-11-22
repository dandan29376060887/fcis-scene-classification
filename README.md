# FCIS 2026 NN — Top-View Scene Classification

Image classifier for the Kaggle competition "FCIS 2026 NN: Top-View Scene
Classification": 11 land-use classes in 256x256 overhead JPEGs, scored on
accuracy (private leaderboard decides the final ranking).

## What's here

```
configs/            one JSON config per from-scratch architecture
scripts/train.py    train + validate + checkpoint one model from a config
scripts/predict.py  write a submission CSV from a trained checkpoint
src/scene_clf/
  config.py         DataConfig / ModelConfig / TrainConfig dataclasses (JSON round-trip)
  constants.py      the 11 class names (exact competition spelling) + label mapping
  seed.py           seeding + device resolution
  data/             datasets, transforms, stratified split, DataLoader factories
  models/           registry + 5 from-scratch architectures (+ optional transfer model)
  training/         Trainer (loop, checkpointing, early stopping, history) + metrics
  inference/        test-time prediction -> submission CSV (optional flip TTA)
tests/              pytest suite (synthetic mini-dataset; no competition data needed)
```

## Rules compliance (checked against the live Rules tab)

- **4+ models from scratch** — `simple_cnn`, `vgg_style`, `residual_cnn`,
  `se_cnn` all define their layers explicitly (conv / BN / pooling /
  dense). Ready-made classes such as `torchvision.models.resnet18` are
  not used for these. `vit_scratch` covers the course update (announced
  by the instructor) that one of the four must be a transformer: its
  attention is built by hand from Linear layers and scaled dot-product
  softmax, not a ready-made transformer class.
- `resnet18_transfer` is included as a legal bonus: pretrained weights
  are explicitly allowed, but it does **not** count toward the four.
- **Neural networks only** for final submissions — every model here is one.
- **No external labeled data** — training uses only `labels.csv` +
  `data/` from the competition. ImageNet normalisation statistics are
  statistics, not data; pretrained weights (transfer model only) are
  allowed by the rules.
- **No hand-labeling of test** — predictions come only from the model.
- **AI assistance** was used to write this code; every team member must
  be able to explain it. The README + docstrings are written for that.
- **Reproducibility** — everything is driven by the JSON config; seeds
  are fixed; `history.json` + `best.pt` + `config.json` per run.
- Teams must be 5-6 members and submissions are capped at 5/day: team
  joining and submissions stay manual decisions on the Kaggle site.

## Setup

```
pip install -r requirements.txt        # torch + torchvision + pillow + numpy
pip install -r requirements-dev.txt    # + pytest, for the test suite
```

On Kaggle Notebooks these are preinstalled; just attach the competition
data and this repo as datasets.

## Data layout

The competition download (`Data/`) already matches what the code expects:

```
Data/
  data/<image_id>.jpg      11,476 labeled training images
  test/<image_id>.jpg      1,100 test images
  labels.csv               image_id,label
  test.csv                 image_id
  sample_submission.csv
```

## Train

```
python scripts/train.py --config configs/simple_cnn.json --data-root Data
python scripts/train.py --config configs/residual_cnn.json --data-root Data
```

Each run writes `runs/<name>/{config.json,best.pt,last.pt,history.json}`.
Validation is a stratified 15% hold-out (seed 42) — same split for every
model, so comparisons are fair.

## Train on Google Colab

Open [`colab_train.ipynb`](colab_train.ipynb) in Google Colab, choose a GPU
runtime, and add `KAGGLE_API_TOKEN` in Colab Secrets. Accept
the competition rules on Kaggle first. The notebook clones this repository,
downloads the competition data directly from Kaggle, and trains the scratch
CNNs and ViT plus a pretrained ResNet-18 bonus. The dataset, checkpoints, and
submission are kept out of this public GitHub repository; run outputs are
saved to Google Drive.

## Predict

```
python scripts/predict.py --config configs/simple_cnn.json \
    --checkpoint runs/simple_cnn/best.pt --data-root Data \
    --out submission.csv --tta
```

The CSV is exactly the scored format: header `image_id,label`, one row
per id in `test.csv` order, class names as in the rules. `--tta`
averages each prediction with its horizontal flip (no extra data used).

## Tests

```
python -m pytest tests/ -q
```

Covers: CSV readers, dataset/label mapping, stratified split integrity,
transform shapes, forward shapes of all five scratch architectures, a
learn-something smoke step, metric correctness, a full Trainer run with
checkpoints, and submission-format validation.

## Honest limits

- Full training needs a GPU (Kaggle's free tier works). CPU is only for
  smoke tests.
- Accuracy numbers only mean something when produced by an actual
  training run; none are claimed until the runs exist.
