#!/usr/bin/env bash
set -euo pipefail

python -m tensor_training_core.module_runner prepare-dataset --config "${1:-configs/experiments/dev_macos.yaml}"
