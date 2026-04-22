# Phase-2 CLI Automation Workflow

## Main Commands

```bash
python -m tensor_training_core.cli dataset import-coco --config <config>
python -m tensor_training_core.cli dataset prepare --config <config>
python -m tensor_training_core.cli train run --config <config>
python -m tensor_training_core.cli train status --job-id <job_id>
python -m tensor_training_core.cli evaluate run --config <config>
python -m tensor_training_core.cli export tflite --config <config>
python -m tensor_training_core.cli export mobile --config <config>
```

## Machine-Readable Output

The CLI emits JSON to stdout for contract-friendly commands. This allows shell automation and CI scripting to parse `job_id`, artifact paths, and summaries directly.

## Recommended Pattern

1. Prepare dataset
2. Train
3. Read status or artifact metadata
4. Export
5. Package mobile bundle
