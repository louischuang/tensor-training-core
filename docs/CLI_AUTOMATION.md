# Phase-2 CLI Automation Workflow

## Main Commands

```bash
python -m tensor_training_core.cli dataset import-coco --config <config>
python -m tensor_training_core.cli dataset prepare --config <config>
python -m tensor_training_core.cli train run --config <config>
python -m tensor_training_core.cli train status --job-id <job_id>
python -m tensor_training_core.cli job retry --job-id <job_id>
python -m tensor_training_core.cli evaluate run --config <config>
python -m tensor_training_core.cli export tflite --config <config>
python -m tensor_training_core.cli export mobile --config <config>
```

## Machine-Readable Output

The CLI emits JSON to stdout for contract-friendly commands. This allows shell automation and CI scripting to parse `job_id`, artifact paths, and summaries directly.

## CLI Correlation Fields

Each CLI invocation now creates a CLI session log under:

- `artifacts/logs/cli/<cli_invocation_id>/application.jsonl`

Structured CLI logs include:

- `cli_invocation_id`
- `cli_command`

When the command triggers a run-scoped operation such as training or export, the same correlation fields are also propagated into the run log so the CLI session and run artifacts can be matched later.

## Recommended Pattern

1. Prepare dataset
2. Train
3. Read status or artifact metadata
4. Retry a failed job when needed
5. Export
6. Package mobile bundle
