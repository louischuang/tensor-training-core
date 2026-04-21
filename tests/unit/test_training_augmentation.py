from __future__ import annotations

import random

import numpy as np

from tensor_training_core.config.schema import AugmentationSettings
from tensor_training_core.training.runner import _augment_image_and_boxes


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
