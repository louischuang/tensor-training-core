# CUDA Docker Training

## Goal

Run TensorFlow-based training, export, and mobile packaging inside the project Docker image.

## Build

```bash
docker build -t tensor-training-core-tf:latest -f docker/Dockerfile.cuda .
```

## Run

```bash
docker run --rm -it -v "$PWD":/workspace -w /workspace tensor-training-core-tf:latest bash
```

Inside the container:

```bash
export PYTHONPATH=src
python -m tensor_training_core.cli dataset prepare --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
python -m tensor_training_core.cli train run --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
python -m tensor_training_core.cli export tflite --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
python -m tensor_training_core.cli export mobile --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

## Smoke Test

```bash
bash tests/integration/run_docker_smoke.sh tensor-training-core-tf:latest
```
