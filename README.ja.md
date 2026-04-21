# Tensor Training Core

言語版: [English](./README.md) | [繁體中文](./README.zh-TW.md) | [简体中文](./README.zh-CN.md) | [日本語](./README.ja.md)

Tensor Training Core は、TensorFlow Lite 向け物体検出学習を目的としたリポジトリです。現在は次の実用的なベースラインに集中しています。

- COCO 形式データセットの取り込み
- MobileNet ベースの物体検出モデル学習
- evaluation とレポート生成
- TensorFlow Lite への変換
- iOS / Android 向けモバイル統合アセット生成
- 共通 Python service layer、CLI、API、agent skill contract

## このリポジトリでできること

同じ workflow を使って、次の環境をサポートすることを目指しています。

- Apple Silicon macOS での開発と検証
- x64 Linux Docker でのより重い学習

現在実装済みの主な流れは次の通りです。

1. COCO 形式データセットを取り込み、検証する
2. internal manifest に正規化する
3. train / val / test manifest に分割する
4. MobileNet ベース detector を学習する
5. checkpoint を評価してレポートを生成する
6. TensorFlow Lite モデルを出力する
7. iOS / Android mobile bundle を生成する
8. 出力した TFLite モデルで inference 検証を行う

## リポジトリ構成

```text
configs/                 データセット、モデル、学習、experiment の YAML 設定
data/                    生データセット、manifest、dataset metadata
docker/                  Docker build / runtime ファイル
scripts/                 実行補助スクリプト
src/tensor_training_core/
  api/                   FastAPI アプリケーションとルート
  config/                設定ローダーと schema
  data/                  データ変換、検証、分割
  evaluation/            評価指標とレポート
  export/                TFLite、SavedModel、labels、mobile export
  inference/             TFLite 検証と preview 画像
  interfaces/            共通 service layer、DTO、job records
  mobile/                Android / iOS bundle writer
  training/              training runner、callbacks、augmentation
artifacts/               生成された experiments、reports、logs、jobs、bundles
tests/                   ユニットテスト
```

## 主要な用語

- `dataset`: 学習に取り込む画像とラベルファイル
- `internal manifest`: システムが生成する、プロジェクト管理下の正規化データ形式
- `job`: `artifacts/jobs/` に保存される実行記録
- `artifact`: `artifacts/experiments/`、`artifacts/reports/`、`artifacts/logs/` に保存される成果物

## 現在実装済みの機能

### Dataset Pipeline

- COCO dataset validation
- annotation quality checks と cleaning report
- manifest generation
- dataset metadata output
- train / val / test split generation

### Training

- MobileNet ベース baseline detector
- TensorFlow training backend
- training logs と per-run artifacts
- pretrained checkpoint loading
- resume-training
- checkpoint と TensorBoard 出力

### Evaluation

- precision / recall / mAP50 指標
- evaluation preview 画像
- `artifacts/reports/` への評価レポート出力

### Export と Mobile Packaging

- SavedModel export
- `float32` / `float16` / `int8` TFLite export
- `label.txt` 生成
- iOS / Android mobile bundle
- integration assumptions と bundle verification ファイル

### 外部インターフェース

- 共通 service layer
- package CLI
- FastAPI skeleton と利用可能な routes
- agent / 外部プラットフォーム向け `SKILL.md`

## クイックスタート

### 1. 依存関係のインストール

ローカル開発:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

Docker 学習:

```bash
docker build -t tensor-training-core-tf:latest -f docker/Dockerfile.cuda .
```

### 2. データセット準備

```bash
python -m tensor_training_core.cli dataset prepare --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

### 3. 学習実行

```bash
python -m tensor_training_core.cli train run --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

### 4. 評価実行

```bash
python -m tensor_training_core.cli evaluate run --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

### 5. TFLite 出力

```bash
python -m tensor_training_core.cli export tflite --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

### 6. モバイルアセット生成

```bash
python -m tensor_training_core.cli export mobile --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

## CLI コマンド

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

利用可能な wrapper scripts:

- `scripts/run_cli.sh`
- `scripts/serve_api.sh`
- `scripts/prepare_coco.sh`
- `scripts/train.sh`
- `scripts/evaluate.sh`
- `scripts/export_tflite.sh`
- `scripts/package_mobile_assets.sh`

## HTTP API

現在の routes:

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

Request 例:

```json
{
  "config_path": "configs/experiments/train_tensorflow_esp32_cam_dev.yaml"
}
```

API 起動:

```bash
python -m tensor_training_core.cli serve api
```

または:

```bash
scripts/serve_api.sh
```

## 重要な出力先

- Jobs: `artifacts/jobs/`
- Experiment runs: `artifacts/experiments/<experiment_id>/<run_id>/`
- Reports: `artifacts/reports/<experiment_id>/<run_id>/`
- Logs: `artifacts/logs/<run_id>/`
- API request log: `artifacts/logs/api/requests.jsonl`

主な成果物:

- checkpoints
- `training_summary.json`
- `evaluation_summary.json`
- evaluation previews
- SavedModel
- `.tflite`
- `label.txt`
- mobile bundle
- structured logs と failure summaries

## 主要ドキュメント

- [Architecture Plan](./ARCHITECTURE.md)
- [TODO List](./TODO.md)
- [Agent Skill Contract](./SKILL.md)

## データセットとライセンスに関する注意

このリポジトリは COCO 形式の取り込みをサポートしていますが、dataset 利用には別途確認が必要です。

- COCO API や関連コードは一般に緩いライセンスです
- COCO dataset 自体には独自の利用条件があります
- 元画像には attribution や redistribution の制約がある場合があります

raw dataset、internal manifest、生成 artifact は分けて管理することを推奨します。

## 上流プロジェクトとライセンス確認

| パッケージ / プロジェクト | この repo での用途 | Repository | 確認したライセンス | 利用メモ |
| --- | --- | --- | --- | --- |
| TensorFlow | コア学習と export フレームワーク | [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow) | Apache-2.0 | 緩いライセンスで、商用・社内利用でも一般的 |
| TensorFlow Models | 物体検出の参照実装 | [tensorflow/models](https://github.com/tensorflow/models) | Apache-2.0 | 今後のモデル整合に有用 |
| KerasCV | 将来拡張用の CV ツール | [keras-team/keras-cv](https://github.com/keras-team/keras-cv) | Apache-2.0 | 将来の拡張候補 |
| TensorFlow Lite Support | mobile metadata と TFLite 補助ツール | [tensorflow/tflite-support](https://github.com/tensorflow/tflite-support) | Apache-2.0 | mobile metadata フローに有用 |
| COCO API / pycocotools | COCO parsing と validation | [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi) | Simplified BSD | データ変換・検証ツールに適している |

## 現在の状態

この repository は、もはや計画書だけではありません。

- baseline Python workflow は動作します
- CLI は利用可能です
- API skeleton は利用可能です
- `SKILL.md` は現在の統合方法を記述しています

今後は CI、ドキュメント拡充、production hardening が中心になります。

## ライセンスに関する注意

この README は技術的な要約であり、法的助言ではありません。製品出荷前には、実際に使う dependency、モデル資産、dataset のライセンスを最終確認してください。
