#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-tensor-training-core-tf:latest}"
WORKDIR="/workspace"
CONFIG="configs/experiments/test_tiny_tensorflow.yaml"

docker run --rm -v "$PWD":"$WORKDIR" -w "$WORKDIR" "$IMAGE_NAME" bash -lc "
set -euo pipefail
export PYTHONPATH=src
python -m tensor_training_core.cli dataset prepare --config $CONFIG >/tmp/prepare.json
python -m tensor_training_core.cli train run --config $CONFIG >/tmp/train.json
python -m tensor_training_core.cli export tflite --config $CONFIG >/tmp/export.json
python -m tensor_training_core.cli export mobile --config $CONFIG >/tmp/mobile.json
python - <<'PY'
import json
from pathlib import Path

prepare = json.loads(Path('/tmp/prepare.json').read_text())
train = json.loads(Path('/tmp/train.json').read_text())
export = json.loads(Path('/tmp/export.json').read_text())
mobile = json.loads(Path('/tmp/mobile.json').read_text())

assert Path(prepare['job']['outputs']['manifest_path']).exists()
assert Path(train['job']['outputs']['checkpoint_path']).exists()
assert Path(export['job']['outputs']['saved_model_dir']).exists()
assert Path(export['job']['outputs']['tflite_path_float32']).exists()
assert Path(export['job']['outputs']['label_txt_path']).exists()
assert Path(export['job']['outputs']['benchmark_report_path']).exists()
assert Path(export['job']['outputs']['model_card_path']).exists()
assert Path(export['job']['outputs']['license_metadata_path']).exists()
assert Path(mobile['job']['outputs']['android_bundle_dir_float32']).exists()
assert Path(mobile['job']['outputs']['ios_bundle_dir_float32']).exists()
assert Path(mobile['job']['outputs']['android_bundle_dir_float32'], 'benchmark_report.json').exists()
assert Path(mobile['job']['outputs']['android_bundle_dir_float32'], 'MODEL_CARD.md').exists()
assert Path(mobile['job']['outputs']['android_bundle_dir_float32'], 'license_metadata.json').exists()
assert Path(mobile['job']['outputs']['ios_bundle_dir_float32'], 'benchmark_report.json').exists()
assert Path(mobile['job']['outputs']['ios_bundle_dir_float32'], 'MODEL_CARD.md').exists()
assert Path(mobile['job']['outputs']['ios_bundle_dir_float32'], 'license_metadata.json').exists()
print('docker smoke ok')
PY
"
