"""Canonical class ordering and label mapping for the competition.

The class names must match the competition specification exactly
(lowercase, underscores) because submission files are scored on them.
"""

CLASSES = [
    "baseball_field",
    "beach",
    "dense_residential",
    "farmland",
    "forest",
    "harbor",
    "medium_residential",
    "parking_lot",
    "river",
    "sparse_residential",
    "storage_tanks",
]

NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

# ImageNet statistics, used for input normalisation. Pretraining on
# ImageNet is allowed by the rules, so these statistics are too.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
