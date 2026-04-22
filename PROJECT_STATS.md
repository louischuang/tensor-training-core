# Project Stats

## Scope

This document tracks a lightweight code-size snapshot for the repository.

Counting rules used in this snapshot:

- focus on `src/` Python implementation code
- exclude generated outputs such as `artifacts/`
- exclude raw datasets such as `data/raw/`
- `code lines` means non-empty lines that are not pure `# ...` comment lines
- this is an engineering estimate, not a compiler-grade LOC metric

## Current Snapshot

### Python Implementation (`src/`)

- files: `45`
- total lines: `3540`
- approximate code lines: `3054`
- blank lines: `486`
- pure `#` comment lines: `0`

### Module Distribution

| Module | Files | Approx Code Lines | Share |
| --- | ---: | ---: | ---: |
| `api` | 6 | 523 | 17.13% |
| `interfaces` | 3 | 521 | 17.06% |
| `training` | 4 | 436 | 14.28% |
| `evaluation` | 2 | 269 | 8.81% |
| `export` | 5 | 268 | 8.78% |
| `inference` | 2 | 246 | 8.05% |
| `data` | 10 | 232 | 7.60% |
| `models` | 3 | 128 | 4.19% |
| `mobile` | 2 | 108 | 3.54% |
| `config` | 2 | 98 | 3.21% |
| `cli.py` | 1 | 96 | 3.14% |
| `utils` | 3 | 90 | 2.95% |
| `module_runner.py` | 1 | 36 | 1.18% |
| `__init__.py` | 1 | 3 | 0.10% |

## Largest Source Files

| File | Approx Code Lines |
| --- | ---: |
| `src/tensor_training_core/interfaces/service.py` | 454 |
| `src/tensor_training_core/training/runner.py` | 398 |
| `src/tensor_training_core/inference/tflite_runner.py` | 211 |
| `src/tensor_training_core/evaluation/evaluator.py` | 170 |
| `src/tensor_training_core/api/routes/training.py` | 163 |
| `src/tensor_training_core/api/schemas.py` | 140 |
| `src/tensor_training_core/export/tflite.py` | 133 |
| `src/tensor_training_core/data/validation.py` | 111 |
| `src/tensor_training_core/evaluation/reports.py` | 99 |
| `src/tensor_training_core/cli.py` | 96 |

## Reading This Snapshot

- The largest implementation areas are `api`, `interfaces`, and `training`.
- The repository is currently centered on orchestration and end-to-end pipeline behavior more than on many separate model families.
- `interfaces/service.py` is currently the biggest consolidation point and is a likely future refactor target if the project keeps growing.
