from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from tensor_training_core.config.schema import DatasetConfig, ModelConfig
from tensor_training_core.interfaces.dto import RunContext
from tensor_training_core.utils.paths import MODELS_DIR, ensure_directory


REGISTRY_VERSION = "1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def build_model_version_descriptor(
    *,
    context: RunContext,
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    export_outputs: dict[str, str],
) -> dict[str, object]:
    export_manifest = json.loads(Path(export_outputs["export_manifest_path"]).read_text(encoding="utf-8"))
    quantizations = sorted(export_manifest["exports"].keys())
    model_key = f"{context.experiment_id}/{model_config.model.name}"
    return {
        "registry_version": REGISTRY_VERSION,
        "model_key": model_key,
        "version_id": context.run_id,
        "registered_at": _utc_now(),
        "experiment_id": context.experiment_id,
        "dataset_version": context.dataset_version,
        "dataset_name": dataset_config.name,
        "model_name": model_config.model.name,
        "model_family": model_config.model.family,
        "quantizations": quantizations,
        "source_run_id": export_manifest["source_run_id"],
        "artifact_dir": str(context.artifact_dir),
        "log_dir": str(context.log_dir),
        "export_manifest_path": export_outputs["export_manifest_path"],
        "label_txt_path": export_outputs["label_txt_path"],
        "model_card_path": export_outputs["model_card_path"],
        "license_metadata_path": export_outputs["license_metadata_path"],
        "benchmark_report_path": export_outputs["benchmark_report_path"],
        "exports": export_manifest["exports"],
    }


def register_model_version(
    *,
    context: RunContext,
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    export_outputs: dict[str, str],
) -> dict[str, str]:
    model_root = ensure_directory(MODELS_DIR / context.experiment_id / model_config.model.name)
    version_dir = ensure_directory(model_root / context.run_id)
    descriptor_path = version_dir / "model_version.json"
    latest_path = model_root / "latest.json"
    model_index_path = model_root / "index.json"
    global_index_path = MODELS_DIR / "index.json"

    descriptor = build_model_version_descriptor(
        context=context,
        model_config=model_config,
        dataset_config=dataset_config,
        export_outputs=export_outputs,
    )
    _write_json(descriptor_path, descriptor)
    _write_json(latest_path, descriptor)

    version_entry = {
        "version_id": context.run_id,
        "registered_at": descriptor["registered_at"],
        "dataset_version": context.dataset_version,
        "descriptor_path": str(descriptor_path),
        "quantizations": descriptor["quantizations"],
    }

    model_index = _load_json(model_index_path) or {
        "registry_version": REGISTRY_VERSION,
        "model_key": descriptor["model_key"],
        "experiment_id": context.experiment_id,
        "model_name": model_config.model.name,
        "versions": [],
    }
    versions = [item for item in model_index.get("versions", []) if item.get("version_id") != context.run_id]
    versions.append(version_entry)
    versions.sort(key=lambda item: item["version_id"], reverse=True)
    model_index["versions"] = versions
    model_index["latest_version_id"] = context.run_id
    model_index["latest_descriptor_path"] = str(latest_path)
    model_index["version_count"] = len(versions)
    _write_json(model_index_path, model_index)

    global_index = _load_json(global_index_path) or {
        "registry_version": REGISTRY_VERSION,
        "models": [],
    }
    models = [item for item in global_index.get("models", []) if item.get("model_key") != descriptor["model_key"]]
    models.append(
        {
            "model_key": descriptor["model_key"],
            "experiment_id": context.experiment_id,
            "model_name": model_config.model.name,
            "latest_version_id": context.run_id,
            "latest_descriptor_path": str(latest_path),
            "version_count": len(versions),
        }
    )
    models.sort(key=lambda item: (item["experiment_id"], item["model_name"]))
    global_index["models"] = models
    _write_json(global_index_path, global_index)

    return {
        "model_registry_dir": str(model_root),
        "model_registry_version_path": str(descriptor_path),
        "model_registry_latest_path": str(latest_path),
        "model_registry_index_path": str(model_index_path),
        "global_model_registry_index_path": str(global_index_path),
    }
