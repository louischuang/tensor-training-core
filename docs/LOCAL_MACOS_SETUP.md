# Local macOS Setup

## Goal

Set up the repository on Apple Silicon macOS for local development, smoke testing, and API work.

## Steps

1. Create and activate a virtual environment.
2. Install the project with development extras.
3. Prepare the sample dataset or your own COCO dataset.
4. Run CLI, API, and tests locally.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

Optional Apple Silicon TensorFlow acceleration can be added later with `tensorflow-metal`, but the repository flow does not require it for baseline development.

## Common Commands

```bash
python -m tensor_training_core.cli dataset prepare --config configs/experiments/dev_macos.yaml
python -m tensor_training_core.cli train run --config configs/experiments/dev_macos.yaml
python -m tensor_training_core.cli serve api
pytest tests/unit tests/integration
```
