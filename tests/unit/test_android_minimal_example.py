from __future__ import annotations

from pathlib import Path


def test_android_minimal_example_files_exist() -> None:
    root = Path("examples/android/minimal-tflite-app")
    expected_files = [
        root / "README.md",
        root / "settings.gradle.kts",
        root / "build.gradle.kts",
        root / "app/build.gradle.kts",
        root / "app/src/main/AndroidManifest.xml",
        root / "app/src/main/assets/README.md",
        root / "app/src/main/java/com/example/tensortrainingcore/MainActivity.kt",
        root / "app/src/main/java/com/example/tensortrainingcore/TfliteDetector.kt",
    ]
    for path in expected_files:
        assert path.exists(), f"Missing Android example file: {path}"


def test_android_minimal_example_readme_mentions_bundle_files() -> None:
    readme = Path("examples/android/minimal-tflite-app/README.md").read_text(encoding="utf-8")
    assert "model.tflite" in readme
    assert "label.txt" in readme
    assert "export_metadata.json" in readme
