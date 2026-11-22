import pytest
import torch

from scene_clf.config import ModelConfig
from scene_clf.models import SCRATCH_MODELS, build_model


@pytest.mark.parametrize("name", SCRATCH_MODELS)
def test_scratch_models_forward_shape(name):
    cfg = ModelConfig(name=name, num_classes=4, dropout=0.1, base_channels=8)
    model = build_model(cfg)
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 4)
    assert model.num_parameters() > 0


def test_models_actually_learn_something_tiny():
    """One optimisation step must reduce the loss on a fixed batch."""
    torch.manual_seed(0)
    model = build_model(ModelConfig(name="simple_cnn", num_classes=4, base_channels=8))
    images = torch.randn(8, 3, 32, 32)
    targets = torch.randint(0, 4, (8,))
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    first = criterion(model(images), targets).item()
    for _ in range(20):
        opt.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        opt.step()
    last = criterion(model(images), targets).item()
    assert last < first


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        build_model(ModelConfig(name="does_not_exist"))
