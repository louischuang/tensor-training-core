from __future__ import annotations

import logging


def build_training_progress_callback(
    tf,
    logger: logging.Logger,
    total_epochs: int,
):
    class TrainingProgressCallback(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None) -> None:
            logger.info("training_started total_epochs=%s", total_epochs)

        def on_epoch_end(self, epoch, logs=None) -> None:
            logs = logs or {}
            metric_parts = [
                f"epoch={epoch + 1}/{total_epochs}",
                f"loss={float(logs.get('loss', 0.0)):.6f}",
            ]
            if "class_output_accuracy" in logs:
                metric_parts.append(f"class_output_accuracy={float(logs['class_output_accuracy']):.6f}")
            if "bbox_output_loss" in logs:
                metric_parts.append(f"bbox_output_loss={float(logs['bbox_output_loss']):.6f}")
            if "class_output_loss" in logs:
                metric_parts.append(f"class_output_loss={float(logs['class_output_loss']):.6f}")
            logger.info("training_epoch_completed %s", " ".join(metric_parts))

        def on_train_end(self, logs=None) -> None:
            logger.info("training_completed")

    return TrainingProgressCallback()
