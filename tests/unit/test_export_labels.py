from __future__ import annotations

from tensor_training_core.export.labels import build_label_lines


def test_build_label_lines_uses_foreground_order() -> None:
    label_map = {
        0: "Cars-Dogs-Houses-Persons-Trees",
        1: "0",
        2: "1",
        3: "2",
        4: "3",
        5: "4",
    }

    assert build_label_lines(label_map) == ["0", "1", "2", "3", "4"]
