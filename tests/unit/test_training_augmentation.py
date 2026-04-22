from __future__ import annotations

import random

import numpy as np

from tensor_training_core.config.schema import AugmentationSettings
from tensor_training_core.training.runner import _augment_image_and_boxes, resolve_augmentation_settings


def test_horizontal_flip_updates_box_coordinates() -> None:
    image = np.zeros((4, 4, 3), dtype=np.float32)
    boxes = [np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)]
    augmentation = AugmentationSettings(
        enabled=True,
        horizontal_flip_prob=1.0,
        brightness_delta=0.0,
        contrast_min=1.0,
        contrast_max=1.0,
    )

    _, augmented_boxes = _augment_image_and_boxes(image, boxes, augmentation, random.Random(42))

    assert np.allclose(augmented_boxes[0], np.asarray([0.6, 0.2, 0.3, 0.4], dtype=np.float32))


def test_augmentation_preset_resolves_standard_values() -> None:
    resolved = resolve_augmentation_settings(AugmentationSettings(preset="standard"))

    assert resolved.enabled is True
    assert resolved.horizontal_flip_prob == 0.5
    assert resolved.brightness_delta == 0.08
    assert resolved.contrast_min == 0.9
    assert resolved.contrast_max == 1.15


def test_augmentation_preset_allows_custom_override() -> None:
    resolved = resolve_augmentation_settings(
        AugmentationSettings(preset="light", brightness_delta=0.2)
    )

    assert resolved.enabled is True
    assert resolved.horizontal_flip_prob == 0.25
    assert resolved.brightness_delta == 0.2
