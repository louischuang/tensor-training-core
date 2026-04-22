from __future__ import annotations

from pathlib import Path


def test_ios_minimal_example_files_exist() -> None:
    root = Path("examples/ios/minimal-tflite-app")
    expected_files = [
        root / "README.md",
        root / "App/Resources/README.md",
        root / "App/TensorTrainingCoreIOSExampleApp.swift",
        root / "App/ContentView.swift",
        root / "App/IOSTfliteDetector.swift",
    ]
    for path in expected_files:
        assert path.exists(), f"Missing iOS example file: {path}"


def test_ios_minimal_example_readme_mentions_bundle_files() -> None:
    readme = Path("examples/ios/minimal-tflite-app/README.md").read_text(encoding="utf-8")
    assert "model.tflite" in readme
    assert "label.txt" in readme
    assert "export_metadata.json" in readme
