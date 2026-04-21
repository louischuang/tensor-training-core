from __future__ import annotations

from dataclasses import dataclass

from tensor_training_core.config.schema import ModelConfig
from tensor_training_core.models.anchors import load_anchor_array


@dataclass(slots=True)
class ModelSpec:
    name: str
    family: str
    image_size: tuple[int, int]
    num_classes: int
    max_detections: int
    anchors: tuple[tuple[float, float, float, float], ...]


def get_model_spec(model_config: ModelConfig) -> ModelSpec:
    return ModelSpec(
        name=model_config.model.name,
        family=model_config.model.family,
        image_size=tuple(model_config.model.image_size),
        num_classes=model_config.model.num_classes,
        max_detections=model_config.model.max_detections,
        anchors=tuple(tuple(float(value) for value in anchor) for anchor in load_anchor_array(model_config).tolist()),
    )


def build_keras_detection_model(model_config: ModelConfig):
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise RuntimeError("TensorFlow is required to build the MobileNet detection model.") from exc

    spec = get_model_spec(model_config)
    if spec.family != "mobilenet":
        raise ValueError(f"Unsupported model family: {spec.family}")

    inputs = tf.keras.Input(shape=(spec.image_size[1], spec.image_size[0], 3), name="image")
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        input_tensor=inputs,
        weights=None,
        alpha=1.0,
    )
    x = backbone.output
    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu", name="det_head_conv")(x)
    x = tf.keras.layers.BatchNormalization(name="det_head_bn")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="det_head_pool")(x)
    x = tf.keras.layers.Dropout(0.2, name="det_head_dropout")(x)

    class_logits = tf.keras.layers.Dense(
        spec.max_detections * (spec.num_classes + 1),
        name="class_logits",
    )(x)
    class_logits = tf.keras.layers.Reshape(
        (spec.max_detections, spec.num_classes + 1),
        name="class_logits_reshape",
    )(class_logits)
    class_output = tf.keras.layers.Softmax(axis=-1, name="class_output")(class_logits)

    bbox_output = tf.keras.layers.Dense(
        spec.max_detections * 4,
        activation="sigmoid",
        name="bbox_dense",
    )(x)
    bbox_output = tf.keras.layers.Reshape(
        (spec.max_detections, 4),
        name="bbox_output",
    )(bbox_output)

    return tf.keras.Model(
        inputs=inputs,
        outputs={"class_output": class_output, "bbox_output": bbox_output},
        name=spec.name,
    )
