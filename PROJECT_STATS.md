# Project Stats

## Scope

This document tracks a lightweight code-size and project-completion snapshot for the repository.

Counting rules used in this snapshot:

- focus on `src/` Python implementation code for the main LOC snapshot
- exclude generated outputs such as `artifacts/`
- exclude raw datasets such as `data/raw/`
- `code lines` means non-empty lines that are not pure `# ...` comment lines
- this is an engineering estimate, not a compiler-grade LOC metric

## Completion Snapshot

### Delivery Status

- `TODO.md` phase items: `16 / 16` phases complete
- `TODO.md` milestone items: first and second integration milestones complete
- priority backlog:
  - `Must Add`: complete
  - `Strongly Recommended`: complete
  - `Nice To Have Later`: complete

### Practical Repository State

The repository now includes:

- end-to-end training, evaluation, export, packaging, and inference verification
- CLI, HTTP API, Swagger, SSE log streaming, and dashboard routes
- model registry, benchmark artifacts, model card, and license metadata
- Android and iOS minimal integration examples
- Docker, CI, setup docs, and third-party integration docs

## Current Snapshot

### Python Implementation (`src/`)

- files: `49`
- total lines: `4391`
- approximate code lines: `3827`
- blank lines: `564`
- pure `#` comment lines: `0`

### Repository By Area

| Area | Files | Lines |
| --- | ---: | ---: |
| `src/` | 54 | 4606 |
| `tests/` | 23 | 1112 |
| `configs/` | 17 | 199 |
| `scripts/` | 9 | 44 |
| `docker/` | 3 | 63 |
| `docs/` | 12 | 684 |
| `examples/` | 16 | 927 |

### File-Type Snapshot

| Type | Files | Lines |
| --- | ---: | ---: |
| `.py` | 67 | 5426 |
| `.md` | 27 | 3586 |
| `.yaml` | 17 | 199 |
| `.yml` | 2 | 80 |
| `.toml` | 1 | 43 |
| `.json` | 7 | 131 |
| `.sh` | 11 | 90 |
| `.kt` | 2 | 346 |
| `.swift` | 3 | 348 |

### Module Distribution (`src/tensor_training_core`)

| Module | Files | Approx Code Lines | Share |
| --- | ---: | ---: | ---: |
| `api` | 7 | 732 | 19.13% |
| `interfaces` | 3 | 604 | 15.78% |
| `export` | 8 | 611 | 15.97% |
| `training` | 4 | 501 | 13.09% |
| `evaluation` | 2 | 269 | 7.03% |
| `inference` | 2 | 246 | 6.43% |
| `data` | 10 | 231 | 6.04% |
| `models` | 3 | 135 | 3.53% |
| `cli.py` | 1 | 136 | 3.55% |
| `mobile` | 2 | 126 | 3.29% |
| `config` | 2 | 99 | 2.59% |
| `utils` | 3 | 98 | 2.56% |
| `module_runner.py` | 1 | 36 | 0.94% |
| `__init__.py` | 1 | 3 | 0.08% |

## Largest Source Files

| File | Approx Code Lines |
| --- | ---: |
| `src/tensor_training_core/interfaces/service.py` | 526 |
| `src/tensor_training_core/training/runner.py` | 463 |
| `src/tensor_training_core/inference/tflite_runner.py` | 211 |
| `src/tensor_training_core/api/routes/training.py` | 196 |
| `src/tensor_training_core/export/tflite.py` | 180 |
| `src/tensor_training_core/evaluation/evaluator.py` | 170 |
| `src/tensor_training_core/api/routes/dashboard.py` | 168 |
| `src/tensor_training_core/api/schemas.py` | 146 |
| `src/tensor_training_core/cli.py` | 136 |
| `src/tensor_training_core/export/registry.py` | 117 |

## Reading This Snapshot

- The project has moved beyond a pure training pipeline and is now split across orchestration, API, export, and deployment-support features.
- The largest implementation areas are now `api`, `interfaces`, `export`, and `training`.
- `interfaces/service.py` remains the main orchestration hub and is still the clearest future refactor target if the repository continues to grow.
- The repository now has a meaningful amount of non-Python delivery material as well, especially `docs/`, `examples/`, and mobile integration assets.
