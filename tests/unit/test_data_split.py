from __future__ import annotations

from tensor_training_core.data.split import split_counts, split_rows


def test_split_counts_preserves_total() -> None:
    train_count, val_count, test_count = split_counts(1313, train_ratio=0.8, val_ratio=0.1)
    assert train_count + val_count + test_count == 1313
    assert train_count == 1050
    assert val_count == 131
    assert test_count == 132


def test_split_rows_is_deterministic() -> None:
    rows = [{"image_path": f"image_{index}.jpg"} for index in range(10)]
    train_rows, val_rows, test_rows = split_rows(rows, train_ratio=0.6, val_ratio=0.2, seed=7)

    assert [row["image_path"] for row in train_rows] == [
        "image_8.jpg",
        "image_3.jpg",
        "image_1.jpg",
        "image_4.jpg",
        "image_7.jpg",
        "image_0.jpg",
    ]
    assert [row["image_path"] for row in val_rows] == ["image_9.jpg", "image_6.jpg"]
    assert [row["image_path"] for row in test_rows] == ["image_2.jpg", "image_5.jpg"]
