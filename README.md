# Tensor Training Core

Tensor Training Core is a planned TensorFlow-based training repository for TensorFlow Lite object detection models.

## Definitions

- `dataset`: images plus labeling files that can be imported into training
- `internal manifest`: the normalized project-owned metadata generated from an imported dataset for training and reproducibility

## Delivery Plan

Phase 1:

- build the workflow as Python modules
- confirm the end-to-end flow and core functionality are complete

Phase 2:

- add CLI
- add HTTP API
- finalize `SKILL.md` for third-party platforms and AI agents

## First Baseline

- MobileNet-based object detection
- COCO dataset import
- TensorFlow Lite export
- deployment packaging for iOS and Android
- cross-platform execution on:
  - macOS Apple Silicon for development
  - x64 Linux + CUDA Docker for heavier training

## Project Goals

- train a MobileNet-based object detection model
- convert COCO-format datasets into a project-owned internal manifest
- export TensorFlow Lite models for mobile usage
- package model assets so iOS and Android apps can integrate them more easily
- keep one shared codebase for macOS development and CUDA Docker training
- add CLI, API, and `SKILL.md` only after the Python module workflow is proven

## Planned Baseline Stack

- TensorFlow 2.x
- TensorFlow Models Object Detection API
- MobileNet-based SSD detector
- TensorFlow Lite export
- TensorFlow Lite Support / metadata tooling for mobile deployment
- COCO import and conversion pipeline

## Planned Workflow

1. Import a COCO dataset into `data/raw/`
2. Validate and normalize annotations
3. Convert the dataset into an internal manifest
4. Build training-ready records
5. Train a MobileNet object detection model
6. Evaluate the trained checkpoint
7. Export to TensorFlow Lite
8. Generate mobile deployment bundles for iOS and Android
9. Run TFLite inference smoke tests
10. After the Python module workflow is validated, expose stable operations through CLI, API, and `SKILL.md`

## Planned External Interfaces

These are phase-2 items and should be added only after the Python module workflow is stable:

- CLI for operators, scripts, and automation pipelines
- HTTP API for third-party platforms and remote orchestration
- `SKILL.md` for AI agents and agent-capable platforms

The intended rule is simple:

- core business logic lives in shared Python service modules
- CLI commands call the shared service modules
- API routes call the same shared service modules
- `SKILL.md` documents how an agent should use the CLI or API safely

## Logging Note

Logging is part of the plan already, but it will be expanded incrementally as each stage becomes real:

- phase-1 module logging for dataset, training, export, and failures
- phase-2 extension for CLI and API correlation fields

## Repository and License Review

The following upstream repositories are good candidates for this project baseline. Based on the currently published repository metadata and license files, these code repositories use permissive licenses that are generally suitable for internal use and commercial projects, subject to normal license compliance requirements such as preserving notices and attribution where required.

| Package / Project | Purpose in this repo | Repository | Observed license | Usage note |
| --- | --- | --- | --- | --- |
| TensorFlow | Core training and export framework | [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow) | Apache-2.0 | Permissive; commonly usable in commercial and internal projects |
| TensorFlow Models | Object Detection API and MobileNet detection configs | [tensorflow/models](https://github.com/tensorflow/models) | Apache-2.0 | Permissive; suitable for baseline training integration |
| KerasCV | Optional future CV utilities and models | [keras-team/keras-cv](https://github.com/keras-team/keras-cv) | Apache-2.0 | Permissive; optional for later expansion |
| TensorFlow Lite Support | TFLite metadata and mobile helper tooling | [tensorflow/tflite-support](https://github.com/tensorflow/tflite-support) | Apache-2.0 | Permissive; useful for iOS / Android packaging and metadata |
| COCO API / pycocotools | Parsing and validating COCO annotations | [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi) | Simplified BSD | Permissive; suitable for dataset conversion tooling |

## Dataset License Note

COCO is appropriate as the first supported dataset format, but dataset usage should be handled carefully:

- the COCO annotation tooling and COCO API are permissively licensed
- the COCO dataset itself has terms of use and the images originate from upstream sources with their own licensing and attribution implications
- for internal experimentation and format support this is usually workable, but redistribution and production dataset usage should be reviewed against the COCO terms and your product's compliance requirements

Because of that, this project will support COCO-format import, while keeping the internal manifest separate from the raw source dataset.

## Planned Output Artifacts

- trained checkpoints
- evaluation reports
- exported `.tflite` model
- label map
- metadata JSON
- iOS mobile bundle
- Android mobile bundle
- logs

## Current Status

The repository currently contains planning documents only:

- [Architecture Plan](./ARCHITECTURE.md)
- [TODO List](./TODO.md)
- [Skill Draft](./SKILL.md)

## Sources Checked

- [TensorFlow GitHub repository](https://github.com/tensorflow/tensorflow)
- [TensorFlow Models GitHub repository](https://github.com/tensorflow/models)
- [KerasCV GitHub repository](https://github.com/keras-team/keras-cv)
- [TensorFlow Lite Support GitHub repository](https://github.com/tensorflow/tflite-support)
- [COCO API GitHub repository](https://github.com/cocodataset/cocoapi)
- [COCO official site](https://cocodataset.org/)

## License Reminder

This README is a planning aid, not legal advice. Before shipping a product, we should still do one final dependency and dataset license review with the exact versions and assets actually used.
