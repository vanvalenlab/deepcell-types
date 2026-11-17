"""Fixed-seed forward smoke tests for the relocated baseline models.

Pins output shape, finiteness, and seed-determinism (NOT exact numeric values,
which would be brittle across torch/CUDA versions). MAPSModel needs only torch;
CellSighterModel needs torchvision (ResNet backbone)."""

import pytest

torch = pytest.importorskip("torch")


def test_maps_model_forward_shape_and_determinism():
    from deepcell_types.baselines.maps.model import MAPSModel

    torch.manual_seed(0)
    model = MAPSModel(input_dim=10, num_classes=3, hidden_dim=16)
    model.eval()
    x = torch.randn(4, 10)
    with torch.no_grad():
        logits, probs = model(x)
    assert tuple(logits.shape) == (4, 3)
    assert tuple(probs.shape) == (4, 3)
    assert torch.isfinite(logits).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)
    torch.manual_seed(0)
    model2 = MAPSModel(input_dim=10, num_classes=3, hidden_dim=16)
    model2.eval()
    with torch.no_grad():
        logits2, _ = model2(x)
    assert torch.allclose(logits, logits2)


def test_cellsighter_model_forward_shape_and_determinism():
    pytest.importorskip("torchvision")
    pytest.importorskip("pandas")  # cellsighter.model -> training.utils imports pandas
    from deepcell_types.baselines.cellsighter.model import CellSighterModel

    torch.manual_seed(0)
    model = CellSighterModel(
        input_channels=6, num_classes=4, model_size="resnet18", pretrained=False
    )
    model.eval()
    x = torch.randn(2, 6, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (2, 4)
    assert torch.isfinite(y).all()
    torch.manual_seed(0)
    model2 = CellSighterModel(
        input_channels=6, num_classes=4, model_size="resnet18", pretrained=False
    )
    model2.eval()
    with torch.no_grad():
        y2 = model2(x)
    assert torch.allclose(y, y2)
