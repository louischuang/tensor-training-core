#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"
python -m uvicorn tensor_training_core.api.app:create_app --factory --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
