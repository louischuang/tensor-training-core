# Tensor Training Core

语言版本：[English](./README.md) | [繁體中文](./README.zh-TW.md) | [简体中文](./README.zh-CN.md) | [日本語](./README.ja.md)

Tensor Training Core 是一个面向 TensorFlow Lite 目标检测训练的项目，聚焦于一条可落地的基线流程：

- 导入 COCO 格式数据集
- 训练 MobileNet 目标检测模型
- 生成 evaluation 与报告
- 导出 TensorFlow Lite
- 生成 iOS / Android 移动端集成资源
- 提供共享 Python service layer、CLI、API 与 agent skill contract

## 这个项目目前能做什么

项目希望使用同一套 workflow 支持：

- Apple Silicon macOS 开发与验证
- x64 Linux Docker 较重训练流程

目前已实现的主流程：

1. 导入并验证 COCO 格式数据集
2. 转换为 internal manifest
3. 切分 train / val / test manifest
4. 训练 MobileNet-based detector
5. 评估 checkpoint 并生成报告
6. 导出 TensorFlow Lite 模型
7. 生成 iOS / Android mobile bundle
8. 对导出的 TFLite 模型进行 inference 验证

## 项目结构

```text
configs/                 数据集、模型、训练、experiment 的 YAML 配置
data/                    原始数据集、manifest 与 dataset metadata
docker/                  Docker build 与 runtime 文件
scripts/                 执行与启动辅助脚本
src/tensor_training_core/
  api/                   FastAPI 应用与路由
  config/                配置加载与 schema
  data/                  数据转换、验证与切分
  evaluation/            评估指标与报告
  export/                TFLite、SavedModel、labels 与 mobile export
  inference/             TFLite 验证与预览图
  interfaces/            共享 service layer、DTO 与 job records
  mobile/                Android 与 iOS bundle writer
  training/              训练 runner、callbacks 与 augmentation
artifacts/               生成的 experiments、reports、logs、jobs 与 bundles
tests/                   单元测试
```

## 核心概念

- `dataset`：可导入训练的图片与标注文件
- `internal manifest`：系统转换后、项目内部持有的标准化数据格式
- `job`：记录在 `artifacts/jobs/` 的执行记录
- `artifact`：记录在 `artifacts/experiments/`、`artifacts/reports/`、`artifacts/logs/` 的产物

## 当前已完成功能

### Dataset Pipeline

- COCO dataset validation
- annotation quality checks 与 cleaning report
- manifest generation
- dataset metadata output
- train / val / test split generation

### Training

- MobileNet-based baseline detector
- TensorFlow training backend
- training logs 与 per-run artifacts
- pretrained checkpoint loading
- resume-training
- checkpoint 与 TensorBoard 输出

### Evaluation

- precision / recall / mAP50 指标
- evaluation preview 图片
- `artifacts/reports/` 评估报告

### Export 与 Mobile Packaging

- SavedModel export
- `float32` / `float16` / `int8` TFLite export
- `label.txt` 生成
- iOS / Android mobile bundle
- integration assumptions 与 bundle verification 文件

### 外部接口

- 共享 service layer
- package CLI
- FastAPI skeleton 与可用 routes
- 面向 agent / 第三方平台的 `SKILL.md`

## 快速开始

### 1. 安装依赖

本地开发：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

Docker 训练：

```bash
docker build -t tensor-training-core-tf:latest -f docker/Dockerfile.cuda .
```

### 2. 准备数据集

```bash
python -m tensor_training_core.cli dataset prepare --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

### 3. 执行训练

```bash
python -m tensor_training_core.cli train run --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

### 4. 执行评估

```bash
python -m tensor_training_core.cli evaluate run --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

### 5. 导出 TFLite

```bash
python -m tensor_training_core.cli export tflite --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

### 6. 生成移动端资源

```bash
python -m tensor_training_core.cli export mobile --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

## CLI 命令

```text
tensor-training-core dataset import-coco --config <path>
tensor-training-core dataset prepare --config <path>
tensor-training-core train run --config <path>
tensor-training-core train status --job-id <id>
tensor-training-core evaluate run --config <path>
tensor-training-core export tflite --config <path>
tensor-training-core export mobile --config <path>
tensor-training-core artifact list --limit <n>
tensor-training-core artifact describe --artifact <path>
tensor-training-core serve api
```

同时提供 wrapper scripts：

- `scripts/run_cli.sh`
- `scripts/serve_api.sh`
- `scripts/prepare_coco.sh`
- `scripts/train.sh`
- `scripts/evaluate.sh`
- `scripts/export_tflite.sh`
- `scripts/package_mobile_assets.sh`

## HTTP API

当前 routes：

```text
GET  /health
POST /datasets/import/coco
POST /datasets/prepare
POST /training/jobs
GET  /training/jobs/{job_id}
POST /exports/tflite
POST /exports/mobile-bundle
GET  /artifacts/{job_id}
```

Request 示例：

```json
{
  "config_path": "configs/experiments/train_tensorflow_esp32_cam_dev.yaml"
}
```

启动 API：

```bash
python -m tensor_training_core.cli serve api
```

或：

```bash
scripts/serve_api.sh
```

## 重要输出位置

- Jobs：`artifacts/jobs/`
- Experiment runs：`artifacts/experiments/<experiment_id>/<run_id>/`
- Reports：`artifacts/reports/<experiment_id>/<run_id>/`
- Logs：`artifacts/logs/<run_id>/`
- API request log：`artifacts/logs/api/requests.jsonl`

常见产物包括：

- checkpoints
- `training_summary.json`
- `evaluation_summary.json`
- evaluation previews
- SavedModel
- `.tflite`
- `label.txt`
- mobile bundle
- structured logs 与 failure summaries

## 重要文档

- [Architecture Plan](./ARCHITECTURE.md)
- [TODO List](./TODO.md)
- [Agent Skill Contract](./SKILL.md)

## 数据集与许可证说明

本项目支持 COCO 格式导入，但 dataset 使用仍需额外审查：

- COCO API 与代码工具本身通常是宽松许可证
- COCO dataset 本身有单独的使用条款
- 原始图片来源可能带有 attribution 与 redistribution 限制

建议将 raw dataset、internal manifest 与生成后的 artifact 明确分开管理。

## 上游项目与许可证检查

| 套件 / 项目 | 在本 repo 中的用途 | Repository | 观察到的许可证 | 使用说明 |
| --- | --- | --- | --- | --- |
| TensorFlow | 核心训练与导出框架 | [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow) | Apache-2.0 | 宽松许可证，常见于商业与内部项目 |
| TensorFlow Models | 目标检测参考实现 | [tensorflow/models](https://github.com/tensorflow/models) | Apache-2.0 | 可作为后续模型对齐参考 |
| KerasCV | 未来可扩展的 CV 工具 | [keras-team/keras-cv](https://github.com/keras-team/keras-cv) | Apache-2.0 | 可作为后续扩展选项 |
| TensorFlow Lite Support | mobile metadata 与 TFLite 工具 | [tensorflow/tflite-support](https://github.com/tensorflow/tflite-support) | Apache-2.0 | 适合 mobile metadata 流程 |
| COCO API / pycocotools | COCO parsing 与 validation | [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi) | Simplified BSD | 适合数据转换与验证工具 |

## 当前状态

这个 repository 已经不只是规划文档：

- baseline Python workflow 已可执行
- CLI 已可用
- API skeleton 已可用
- `SKILL.md` 已可描述当前实际集成方式

后续重点会更多放在 CI、文档深化与 production hardening。

## 许可证提醒

这份 README 是工程说明，不是法律意见。真正要交付产品前，仍然要根据实际使用的 dependency、模型资源与 dataset 做最后一次许可证检查。
