import sys

import pytest
import torch

from model import MAFNet


def _maf_module():
    # model/__init__.py binds the name MAFNet to the class, shadowing the module.
    return sys.modules["model.MAFNet"]


def test_resnet34_weight_enum_member_exists():
    from torchvision import models

    if not hasattr(models, "ResNet34_Weights"):
        pytest.skip("torchvision has no ResNet34_Weights")
    assert hasattr(models.ResNet34_Weights, "IMAGENET1K_V1")


def test_resnet34_pretrained_selects_weights(monkeypatch):
    maf = _maf_module()

    calls = []

    def fake_resnet34(*args, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(maf.models, "resnet34", fake_resnet34)
    monkeypatch.setattr(
        maf.models,
        "ResNet34_Weights",
        type("W", (), {"IMAGENET1K_V1": "IMAGENET"}),
        raising=False,
    )

    assert maf._resnet34(True) is not None
    assert maf._resnet34(False) is not None
    assert maf._resnet34() is not None
    assert calls == [
        {"weights": "IMAGENET"},
        {"weights": None},
        {"weights": "IMAGENET"},
    ]


def test_resnet34_fallback_pretrained_flag(monkeypatch):
    maf = _maf_module()

    monkeypatch.delattr(maf.models, "ResNet34_Weights", raising=False)
    calls = []

    def fake_resnet34(*args, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(maf.models, "resnet34", fake_resnet34)
    assert maf._resnet34(True) is not None
    assert maf._resnet34(False) is not None
    assert calls == [{"pretrained": True}, {"pretrained": False}]


def test_mafnet_forwards_pretrained_flag(monkeypatch):
    maf = _maf_module()

    seen = []

    def fake(pretrained=True):
        seen.append(pretrained)
        return object()

    monkeypatch.setattr(maf, "_resnet34", fake)
    # Both backbones are constructed before any attribute of the stub is used.
    with pytest.raises(AttributeError):
        maf.MAFNet(n_class=2, input_h=32, input_w=32, pretrained=False)
    assert seen == [False, False]

    seen.clear()
    with pytest.raises(AttributeError):
        maf.MAFNet(n_class=2, input_h=32, input_w=32)
    assert seen == [True, True]


def test_mafnet_forward_backward_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = MAFNet(n_class=2, input_h=64, input_w=64, pretrained=False)
    model.eval()
    rgb = torch.randn(1, 3, 64, 64)
    disparity = torch.randn(1, 3, 64, 64)
    out = model(rgb, disparity)
    assert out.shape == (1, 2, 64, 64)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert any(p.grad is not None for p in model.parameters())
