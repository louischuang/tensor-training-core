from __future__ import annotations

import sys
import types

from tensor_training_core.config.loader import load_model_config
from tensor_training_core.models.factory import build_keras_detection_model


class _FakeLayer:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, x):
        return f"{self.name}({x})"


class _FakeDense(_FakeLayer):
    def __init__(self, units: int, activation: str | None = None, name: str = "dense") -> None:
        super().__init__(name)
        self.units = units
        self.activation = activation


class _FakeReshape(_FakeLayer):
    def __init__(self, target_shape: tuple[int, ...], name: str = "reshape") -> None:
        super().__init__(name)
        self.target_shape = target_shape


class _FakeSoftmax(_FakeLayer):
    def __init__(self, axis: int = -1, name: str = "softmax") -> None:
        super().__init__(name)
        self.axis = axis


class _FakeDropout(_FakeLayer):
    def __init__(self, rate: float, name: str = "dropout") -> None:
        super().__init__(name)
        self.rate = rate


class _FakeApplications:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def MobileNetV2(self, **kwargs):
        self.calls.append(("mobilenet", kwargs))
        return types.SimpleNamespace(output="mobilenet_output")

    def EfficientNetB0(self, **kwargs):
        self.calls.append(("efficientnet", kwargs))
        return types.SimpleNamespace(output="efficientnet_output")


class _FakeModel:
    def __init__(self, *, inputs, outputs, name: str) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.name = name


def _install_fake_tensorflow(monkeypatch):
    applications = _FakeApplications()
    fake_tf = types.SimpleNamespace(
        keras=types.SimpleNamespace(
            Input=lambda shape, name=None: {"shape": shape, "name": name},
            Model=_FakeModel,
            applications=applications,
            layers=types.SimpleNamespace(
                Conv2D=lambda *args, **kwargs: _FakeLayer(kwargs.get("name", "conv2d")),
                BatchNormalization=lambda *args, **kwargs: _FakeLayer(kwargs.get("name", "batch_norm")),
                GlobalAveragePooling2D=lambda *args, **kwargs: _FakeLayer(kwargs.get("name", "gap")),
                Dropout=lambda *args, **kwargs: _FakeDropout(*args, **kwargs),
                Dense=lambda *args, **kwargs: _FakeDense(*args, **kwargs),
                Reshape=lambda *args, **kwargs: _FakeReshape(*args, **kwargs),
                Softmax=lambda *args, **kwargs: _FakeSoftmax(*args, **kwargs),
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    return applications


def test_build_model_uses_mobilenet_backbone(monkeypatch) -> None:
    applications = _install_fake_tensorflow(monkeypatch)
    model_config = load_model_config("configs/models/ssd_mobilenet_v2_fpnlite_320.yaml")

    model = build_keras_detection_model(model_config)

    assert model.name == "ssd_mobilenet_v2_fpnlite_320x320"
    assert applications.calls[0][0] == "mobilenet"


def test_build_model_supports_efficientnet_family(monkeypatch) -> None:
    applications = _install_fake_tensorflow(monkeypatch)
    model_config = load_model_config("configs/models/detector_efficientnet_b0_320.yaml")

    model = build_keras_detection_model(model_config)

    assert model.name == "detector_efficientnet_b0_320x320"
    assert applications.calls[0][0] == "efficientnet"
