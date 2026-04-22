# CUDA Docker Training

## Goal

Run TensorFlow-based training, export, and mobile packaging inside an x86_64 Linux Docker image with NVIDIA GPU support.

## Host Requirements

- Linux x86_64 host
- NVIDIA GPU with a working host driver
- Docker Engine 19.03+ or newer
- NVIDIA Container Toolkit installed and configured for Docker

Official references:

- [TensorFlow Docker GPU guide](https://www.tensorflow.org/install/docker)
- [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Before using this repository image, verify host GPU passthrough first:

```bash
docker run --rm --gpus all tensorflow/tensorflow:latest-gpu \
  python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Build

```bash
docker build -t tensor-training-core-tf:latest -f docker/Dockerfile.cuda .
```

Or with Docker Compose:

```bash
docker compose -f docker/docker-compose.cuda.yml build
```

## Run

```bash
docker run --rm -it --gpus all -v "$PWD":/workspace -w /workspace \
  -e PYTHONPATH=/workspace/src \
  -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  tensor-training-core-tf:latest bash
```

Or with Docker Compose:

```bash
docker compose -f docker/docker-compose.cuda.yml run --rm trainer
```

Quick GPU check inside the container:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Inside the container:

```bash
python -m tensor_training_core.cli dataset prepare --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
python -m tensor_training_core.cli train run --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
python -m tensor_training_core.cli export tflite --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
python -m tensor_training_core.cli export mobile --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

## Compose Notes

The compose service is configured with:

- `platform: linux/amd64`
- `gpus: all`
- `NVIDIA_VISIBLE_DEVICES=all`
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility`
- `TF_FORCE_GPU_ALLOW_GROWTH=true`

If you need a different TensorFlow GPU base image, override:

```bash
TF_GPU_IMAGE=tensorflow/tensorflow:latest-gpu docker compose -f docker/docker-compose.cuda.yml build
```

## Smoke Test

```bash
bash tests/integration/run_docker_smoke.sh tensor-training-core-tf:latest
```
