# Tensor Training Core

語言版本：[English](./README.md) | [繁體中文](./README.zh-TW.md) | [简体中文](./README.zh-CN.md) | [日本語](./README.ja.md)

Tensor Training Core 是一個以 TensorFlow Lite 物件偵測訓練為核心的專案，聚焦在一條可落地的基線流程：

- 匯入 COCO 格式資料集
- 訓練 MobileNet 物件偵測模型
- 產生 evaluation 與報表
- 匯出 TensorFlow Lite
- 產生 iOS / Android 手機整合資產
- 提供共用 Python service layer、CLI、API 與 agent skill contract

## 這個專案目前能做什麼

此專案希望用同一套 workflow 支援：

- Apple Silicon macOS 開發與驗證
- x64 Linux Docker 較重的訓練流程

目前已實作的主流程是：

1. 匯入並驗證 COCO 格式資料集
2. 轉成 internal manifest
3. 切成 train / val / test manifest
4. 訓練 MobileNet-based detector
5. 評估 checkpoint 並產生報告
6. 匯出 TensorFlow Lite 模型
7. 產生 iOS / Android mobile bundle
8. 對匯出的 TFLite 模型做 inference 驗證

## 專案結構

```text
configs/                 資料集、模型、訓練、experiment 的 YAML 設定
data/                    原始資料集、manifest 與 dataset metadata
docker/                  Docker build 與 runtime 檔案
scripts/                 執行與啟動輔助腳本
src/tensor_training_core/
  api/                   FastAPI 應用與路由
  config/                設定載入與 schema
  data/                  資料轉換、驗證與切分
  evaluation/            評估指標與報表
  export/                TFLite、SavedModel、labels 與 mobile export
  inference/             TFLite 驗證與預覽圖
  interfaces/            共用 service layer、DTO 與 job records
  mobile/                Android 與 iOS bundle writer
  training/              訓練 runner、callbacks 與 augmentation
artifacts/               產生出的 experiments、reports、logs、jobs 與 bundles
tests/                   單元測試
```

## 核心名詞

- `dataset`：可匯入訓練的圖片與標註檔
- `internal manifest`：系統轉換後、專案內部持有的標準化資料格式
- `job`：記錄在 `artifacts/jobs/` 的執行紀錄
- `artifact`：記錄在 `artifacts/experiments/`、`artifacts/reports/`、`artifacts/logs/` 的產物

## 目前已完成功能

### Dataset Pipeline

- COCO dataset validation
- annotation quality checks 與 cleaning report
- manifest generation
- dataset metadata output
- train / val / test split generation

### Training

- MobileNet-based baseline detector
- TensorFlow training backend
- training logs 與 per-run artifacts
- pretrained checkpoint loading
- resume-training
- checkpoint 與 TensorBoard 輸出

### Evaluation

- precision / recall / mAP50 指標
- evaluation preview 圖片
- `artifacts/reports/` 評估報表

### Export 與 Mobile Packaging

- SavedModel export
- `float32` / `float16` / `int8` TFLite export
- `label.txt` 產生
- iOS / Android mobile bundle
- integration assumptions 與 bundle verification 檔案

### 外部介面

- 共用 service layer
- package CLI
- FastAPI skeleton 與可用 routes
- 提供給 agent / 第三方平台的 `SKILL.md`

## 快速開始

### 1. 安裝依賴

本機開發：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

Docker 訓練：

```bash
docker build -t tensor-training-core-tf:latest -f docker/Dockerfile.cuda .
```

### 2. 準備資料集

```bash
python -m tensor_training_core.cli dataset prepare --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

### 3. 執行訓練

```bash
python -m tensor_training_core.cli train run --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

### 4. 執行評估

```bash
python -m tensor_training_core.cli evaluate run --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

### 5. 匯出 TFLite

```bash
python -m tensor_training_core.cli export tflite --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

### 6. 產生手機資產

```bash
python -m tensor_training_core.cli export mobile --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

## CLI 指令

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

也提供 wrapper scripts：

- `scripts/run_cli.sh`
- `scripts/serve_api.sh`
- `scripts/prepare_coco.sh`
- `scripts/train.sh`
- `scripts/evaluate.sh`
- `scripts/export_tflite.sh`
- `scripts/package_mobile_assets.sh`

## HTTP API

目前 routes：

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

Request 範例：

```json
{
  "config_path": "configs/experiments/train_tensorflow_esp32_cam_dev.yaml"
}
```

啟動 API：

```bash
python -m tensor_training_core.cli serve api
```

或：

```bash
scripts/serve_api.sh
```

## 重要輸出位置

- Jobs：`artifacts/jobs/`
- Experiment runs：`artifacts/experiments/<experiment_id>/<run_id>/`
- Reports：`artifacts/reports/<experiment_id>/<run_id>/`
- Logs：`artifacts/logs/<run_id>/`
- API request log：`artifacts/logs/api/requests.jsonl`

常見產物包含：

- checkpoints
- `training_summary.json`
- `evaluation_summary.json`
- evaluation previews
- SavedModel
- `.tflite`
- `label.txt`
- mobile bundle
- structured logs 與 failure summaries

## 重要文件

- [Architecture Plan](./ARCHITECTURE.md)
- [TODO List](./TODO.md)
- [Agent Skill Contract](./SKILL.md)

## 資料集與授權說明

本專案支援 COCO 格式匯入，但 dataset 使用仍要額外審查：

- COCO API 與程式工具本身多半是寬鬆授權
- COCO dataset 本身有自己的使用條款
- 原始圖片來源可能有 attribution 與 redistribution 限制

建議 raw dataset、internal manifest、產生後 artifact 明確分開管理。

## 上游專案與授權檢查

| 套件 / 專案 | 在本 repo 的用途 | Repository | 觀察到的授權 | 使用說明 |
| --- | --- | --- | --- | --- |
| TensorFlow | 核心訓練與匯出框架 | [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow) | Apache-2.0 | 寬鬆授權，常見於商業與內部專案 |
| TensorFlow Models | 物件偵測參考實作 | [tensorflow/models](https://github.com/tensorflow/models) | Apache-2.0 | 可作為後續模型對齊參考 |
| KerasCV | 未來可擴充的 CV 工具 | [keras-team/keras-cv](https://github.com/keras-team/keras-cv) | Apache-2.0 | 可作為後續擴充選項 |
| TensorFlow Lite Support | mobile metadata 與 TFLite 工具 | [tensorflow/tflite-support](https://github.com/tensorflow/tflite-support) | Apache-2.0 | 適合 mobile metadata 流程 |
| COCO API / pycocotools | COCO parsing 與 validation | [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi) | Simplified BSD | 適合資料轉換與驗證工具 |

## 目前狀態

這個 repository 已經不是只有規劃文件：

- baseline Python workflow 已可執行
- CLI 已可用
- API skeleton 已可用
- `SKILL.md` 已能描述現在的實際整合方式

後續重點會更偏向 CI、文件深化與 production hardening。

## 授權提醒

這份 README 是工程說明，不是法律意見。真正要出貨前，仍然要依照實際使用的 dependency、模型資產與 dataset 做最後一次授權檢查。
